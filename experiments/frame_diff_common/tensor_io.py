from pathlib import Path
import torch


def to_torch(tensor):
    if isinstance(tensor, (list, tuple)):
        return [to_torch(t) for t in tensor]
    if hasattr(tensor, "asnumpy"):
        return torch.from_numpy(tensor.asnumpy())
    return tensor


def save_tensor(tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = to_torch(tensor)
    if isinstance(t, (list, tuple)):
        filtered = [x.cpu() for x in t if x is not None]
        if len(filtered) == 1:
            torch.save(filtered[0], path)
        else:
            torch.save(filtered, path)
    else:
        torch.save(t.cpu(), path)


def load_tensor(path: Path, map_location="cpu", weights_only=True):
    t = torch.load(path, map_location=map_location, weights_only=weights_only)
    return t


def to_ms(tensor):
    import mindspore as ms
    if isinstance(tensor, (list, tuple)):
        return [to_ms(t) for t in tensor]
    if hasattr(tensor, "asnumpy"):
        return tensor
    return ms.Tensor(tensor.numpy())
