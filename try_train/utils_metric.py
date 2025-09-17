import torch
from typing import Dict, Iterable, List

def _idcg_at_k(m: int, k: int) -> float:
    upto = min(m, k)
    if upto == 0:
        return 0.0
    denom = torch.log2(torch.arange(2, 2 + upto, dtype=torch.float32))
    return float((1.0 / denom).sum().item())

def _metrics_one(r: torch.Tensor, g: torch.Tensor, k: int, metrics: List[str]) -> Dict[str, float]:
    """
    r: [nbeam] ranked item ids (best -> worst)
    g: [num_gt] ground-truth item ids (unique)
    """
    m = int(g.numel())
    if m == 0 or k <= 0:
        return {name: 0.0 for name in metrics}

    topk = r[:k]
    rel = torch.isin(topk, g)          # [k] bool
    rel_f = rel.float()

    results = {}

    if "HR" in metrics:
        results["HR"] = float(rel.any().item())

    if "Recall" in metrics:
        results["Recall"] = float(rel_f.sum().item() / m)

    if "Precision" in metrics:
        results["Precision"] = float(rel_f.mean().item())

    if "NDCG" in metrics:
        denom = torch.log2(torch.arange(2, 2 + topk.numel(), dtype=torch.float32))
        dcg = float((rel_f / denom).sum().item())
        idcg = _idcg_at_k(m, k)
        results["NDCG"] = float(dcg / idcg) if idcg > 0 else 0.0

    if "MRR" in metrics:
        rel_full = torch.isin(r, g)
        if rel_full.any():
            first_idx = int(torch.nonzero(rel_full, as_tuple=False)[0].item())  # 0-based
            results["MRR"] = 1.0 / (first_idx + 1)
        else:
            results["MRR"] = 0.0

    if "MAP" in metrics:
        cumsum_rel = torch.cumsum(rel_f, dim=0)
        ranks = torch.arange(1, topk.numel() + 1, dtype=torch.float32)
        precision_at_i = torch.where(rel, cumsum_rel / ranks, torch.zeros_like(cumsum_rel))
        ap_sum = float(precision_at_i.sum().item())
        results["MAP"] = ap_sum / float(min(m, k))

    return results

def eval_from_beams(
    ranked: torch.Tensor,        # [B, nbeam] long
    labels: torch.Tensor,        # [B, seqlen] long, with -100 to ignore
    ks: Iterable[int] = [1, 5, 10],
    ignore_index: int = -100,
    metrics: List[str] = ["HR", "Recall", "Precision", "NDCG", "MRR", "MAP"],
) -> Dict[str, Dict[int, float]]:
    """
    Returns micro-averaged metrics across the batch for each K.
    Converts each labels[i] row into a set of ground-truth item ids, ignoring `ignore_index`.
    """
    assert ranked.dim() == 2 and labels.dim() == 2, "ranked=[B,nbeam], labels=[B,seqlen]"
    B, nbeam = ranked.size()
    ranked = ranked.long()
    labels = labels.long()

    # Build per-sample ground-truth sets
    gt_list = []
    for i in range(B):
        mask = labels[i] != ignore_index
        g = labels[i][mask]
        if g.numel() > 0:
            g = g.unique()
        gt_list.append(g)

    # Aggregate
    out = {name: {k: 0.0 for k in ks} for name in metrics}
    for i in range(B):
        r = ranked[i]
        g = gt_list[i]
        for k in ks:
            m = _metrics_one(r, g, k, metrics)
            for name in out:
                out[name][k] += m[name]
    return out
