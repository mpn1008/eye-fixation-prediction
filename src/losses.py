import torch
import torch.nn.functional as F
import numpy as np

_EPS = 1e-7

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_prob(x: torch.Tensor) -> torch.Tensor:
    """Normalise so the spatial sum = 1 (treat map as a probability distribution)."""
    b = x.shape[0]
    flat = x.view(b, -1)
    s = flat.sum(dim=1, keepdim=True).view(b, 1, 1, 1)
    return x / (s + _EPS)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def kl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL divergence: KL(target || pred), both normalised to probability distributions."""
    p = _to_prob(target)
    q = _to_prob(pred)
    return (p * torch.log(p / (q + _EPS) + _EPS)).sum(dim=[1, 2, 3]).mean()


def cc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson CC so it can be minimised."""
    b = pred.shape[0]
    p = pred.view(b, -1)
    t = target.view(b, -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    denom = (p.norm(dim=1) * t.norm(dim=1)).clamp(min=_EPS)
    return (1.0 - (num / denom)).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


class SaliencyLoss(torch.nn.Module):
    def __init__(self, w_kl: float = 1.0, w_cc: float = 1.0, w_mse: float = 0.5):
        super().__init__()
        self.w_kl = w_kl
        self.w_cc = w_cc
        self.w_mse = w_mse

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_kl = kl_loss(pred, target)
        l_cc = cc_loss(pred, target)
        l_mse = mse_loss(pred, target)
        total = self.w_kl * l_kl + self.w_cc * l_cc + self.w_mse * l_mse
        breakdown = {
            "loss/kl": l_kl.item(),
            "loss/cc": l_cc.item(),
            "loss/mse": l_mse.item(),
            "loss/total": total.item(),
        }
        return total, breakdown


# ---------------------------------------------------------------------------
# Metrics  (no_grad context assumed at call site)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute KL, CC and AUC-Judd for a batch. Returns Python floats."""
    b = pred.shape[0]

    # KL
    kl = kl_loss(pred, target).item()

    # CC
    p = pred.view(b, -1)
    t = target.view(b, -1)
    p_c = p - p.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)
    num = (p_c * t_c).sum(dim=1)
    denom = (p_c.norm(dim=1) * t_c.norm(dim=1)).clamp(min=_EPS)
    cc = (num / denom).mean().item()

    # AUC-Judd – per-image, averaged over batch
    target_np = target.cpu().numpy()  # (B, 1, H, W)
    pred_np = pred.cpu().numpy()  # (B, 1, H, W)

    auc_per = []
    for i in range(b):
        sal = pred_np[i, 0]  # (H, W)
        fix = target_np[i, 0]  # (H, W)

        # 1. Normalise saliency map to [0, 1]
        s_min, s_max = sal.min(), sal.max()
        if s_max - s_min > 0:
            sal = (sal - s_min) / (s_max - s_min)
        else:
            auc_per.append(0.5)
            continue

        # 2. Saliency values at fixation locations and all pixels
        fix_sal = sal[fix > 0]
        if len(fix_sal) == 0:
            continue
        all_sal = sal.ravel()
        num_fix = len(fix_sal)
        num_pix = len(all_sal)

        # 3. Vectorised ROC curve via searchsorted – O(N log N) instead of O(Nfix * N)
        thresholds = np.sort(fix_sal)[::-1]  # descending
        all_sal_sorted = np.sort(all_sal)  # ascending, for searchsorted

        # Number of all pixels >= each threshold = N - first index where all_sal >= thresh
        above_counts = num_pix - np.searchsorted(all_sal_sorted, thresholds, side="left")

        j = np.arange(1, num_fix + 1, dtype=np.float64)
        tpr = np.concatenate([[0.0], j / num_fix, [1.0]])
        fpr = np.concatenate([[0.0], above_counts / num_pix, [1.0]])
        auc_per.append(float(np.trapezoid(tpr, fpr)))

    auc_judd = float(sum(auc_per) / len(auc_per)) if auc_per else 0.0

    return {"metric/kl": kl, "metric/cc": cc, "metric/auc_judd": auc_judd}
