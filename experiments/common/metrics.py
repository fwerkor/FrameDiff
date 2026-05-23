import torch


def compute_diff_metrics(output_a: torch.Tensor, output_b: torch.Tensor, eps: float = 1e-8) -> dict:
    eps = float(eps)
    diff = torch.abs(output_a - output_b)
    return {
        "max_abs_diff": float(torch.max(diff)),
        "mean_abs_diff": float(torch.mean(diff)),
        "mean_rel_err": float(torch.mean(diff / (torch.abs(output_a) + eps))),
        "max_rel_err": float(torch.max(diff / (torch.abs(output_a) + eps))),
        "l2_norm": float(torch.norm(output_a - output_b, p=2)),
    }


def compute_metamorphic_metrics(baseline: torch.Tensor, perturbed: torch.Tensor, eps: float = 1e-8) -> dict:
    eps = float(eps)
    delta = torch.abs(perturbed - baseline)
    return {
        "max_abs_delta": float(torch.max(delta)),
        "mean_abs_delta": float(torch.mean(delta)),
        "rel_delta_max": float(torch.max(delta / (torch.abs(baseline) + eps))),
        "rel_delta_mean": float(torch.mean(delta / (torch.abs(baseline) + eps))),
    }
