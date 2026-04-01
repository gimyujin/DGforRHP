from tqdm import tqdm
from torch.func import functional_call

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "2"  # Set the GPU 2 to use

def _forward_logits(model, batch, device, use_amp: bool):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
        if "num" in batch:
            num = batch["num"].to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, num=num)
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
    return logits


def _forward_logits_functional(model, params, buffers, batch, device, use_amp: bool):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)

    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "num" in batch:
        kwargs["num"] = batch["num"].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
        logits = functional_call(model, (params, buffers), args=(), kwargs=kwargs)  # 
    return logits



def train_one_epoch_mldg_amp(
    model,
    domain_loaders: Dict[str, torch.utils.data.DataLoader],
    optimizer,
    device,
    scaler_amp: Optional[torch.cuda.amp.GradScaler],
    inner_lr: float = 1e-3,
    lam: float = 1.0,
    max_grad_norm: float = 1.0,
    steps_per_epoch: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    use_amp = scaler_amp is not None and scaler_amp.is_enabled()

    domains: List[str] = list(domain_loaders.keys())
    if len(domains) < 2:
        raise ValueError("MLDG requires at least 2 domains to split into Support and Query.")

    iters = {d: iter(domain_loaders[d]) for d in domains}
    if steps_per_epoch is None:
        steps_per_epoch = min(len(domain_loaders[d]) for d in domains)

    rng = random.Random(seed)

    total = 0
    running_loss = 0.0
    running_loss_s = 0.0
    running_loss_q = 0.0
    correct = 0

    for _ in tqdm(range(steps_per_epoch), desc="MLDG Train", leave=False):
        # 1. 도메인 분할 (Multi-source / Single-target)
        shuffled_domains = domains.copy()
        rng.shuffle(shuffled_domains)
        
        d_q = shuffled_domains[-1]          # 마지막 1개를 메타 테스트로
        d_s_list = shuffled_domains[:-1]    # 나머지를 모두 메타 소스로
        
        # 2. 배치를 가져오는 함수
        def get_batch(d_name):
            try:
                return next(iters[d_name])
            except StopIteration:
                iters[d_name] = iter(domain_loaders[d_name])
                return next(iters[d_name])

        # Support 도메인들에서 배치 및 라벨 준비
        batches_s = [get_batch(d) for d in d_s_list]
        batch_q = get_batch(d_q)
        labels_q = batch_q["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ---------------------------------------------------------
        # (A) Support forward (Multi-source average)
        # ---------------------------------------------------------
        loss_s_fast_list = []
        for b_s in batches_s:
            l_s = b_s["labels"].to(device, non_blocking=True)
            logits_s = _forward_logits(model, b_s, device, use_amp)
            loss_s_fast_list.append(loss_fn(logits_s, l_s).float())
        
        # Support 도메인들의 평균 loss 계산
        loss_s_fast = torch.stack(loss_s_fast_list).mean()

        # (B) Inner update -> fast params
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        buffers = dict(model.named_buffers())

        grads = torch.autograd.grad(
            loss_s_fast,
            list(params.values()),
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )

        fast_params = {}
        for (name, p), g in zip(params.items(), grads):
            fast_params[name] = p if g is None else (p - inner_lr * g)

        # ---------------------------------------------------------
        # (C) Query forward with fast weights
        # ---------------------------------------------------------
        logits_q = _forward_logits_functional(model, fast_params, buffers, batch_q, device, use_amp)
        loss_q = loss_fn(logits_q, labels_q).float()

        # (D) Outer update를 위한 Support loss 재계산 (Multi-source)
        loss_s_list = []
        for b_s in batches_s:
            l_s = b_s["labels"].to(device, non_blocking=True)
            logits_s2 = _forward_logits(model, b_s, device, use_amp)
            loss_s_list.append(loss_fn(logits_s2, l_s).float())
        
        loss_s = torch.stack(loss_s_list).mean()

        # 최종 Meta-loss
        loss = loss_s + lam * loss_q

        # ---------------------------------------------------------
        # (E) Outer update
        # ---------------------------------------------------------
        if use_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # logging
        bs_q = labels_q.size(0)
        total += bs_q
        running_loss += loss.item() * bs_q
        running_loss_s += loss_s.item() * bs_q
        running_loss_q += loss_q.item() * bs_q
        preds = torch.argmax(logits_q.detach(), dim=1)
        correct += (preds == labels_q).sum().item()

    return {
        "loss": running_loss / max(total, 1),
        "loss_s": running_loss_s / max(total, 1),
        "loss_q": running_loss_q / max(total, 1),
        "acc_q": correct / max(total, 1),
        "steps": float(steps_per_epoch),
    }