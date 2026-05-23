import torch


def compute_diff_metrics(output_a: torch.Tensor, output_b: torch.Tensor, eps: float = 1e-8) -> dict:
    eps = float(eps)
    # Cast to float for integer/boolean tensor support
    a = output_a.float()
    b = output_b.float()
    diff = torch.abs(a - b)
    return {
        "max_abs_diff": float(torch.max(diff)),
        "mean_abs_diff": float(torch.mean(diff)),
        "mean_rel_err": float(torch.mean(diff / (torch.abs(a) + eps))),
        "max_rel_err": float(torch.max(diff / (torch.abs(a) + eps))),
        "l2_norm": float(torch.norm(a - b, p=2)),
    }


def compute_metamorphic_metrics(baseline: torch.Tensor, perturbed: torch.Tensor, eps: float = 1e-8) -> dict:
    eps = float(eps)
    # Cast to float for integer/boolean tensor support
    bl = baseline.float()
    pt = perturbed.float()
    delta = torch.abs(pt - bl)
    return {
        "max_abs_delta": float(torch.max(delta)),
        "mean_abs_delta": float(torch.mean(delta)),
        "rel_delta_max": float(torch.max(delta / (torch.abs(bl) + eps))),
        "rel_delta_mean": float(torch.mean(delta / (torch.abs(bl) + eps))),
    }
