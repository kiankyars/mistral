import torch


def _orthogonalize_gradient(g: torch.Tensor, steps: int = 2, eps: float = 1e-7) -> torch.Tensor:
    """
    Muon-style gradient shaping:
    - Normalize matrix gradients
    - Apply Newton-Schulz iterations toward orthogonality
    """
    if g.ndim < 2:
        return g

    original_shape = g.shape
    mat = g.reshape(g.shape[0], -1)
    transposed = mat.shape[0] < mat.shape[1]
    if transposed:
        mat = mat.t()

    mat = mat / (mat.norm() + eps)
    for _ in range(steps):
        mat = 1.5 * mat - 0.5 * mat @ (mat.t() @ mat)

    if transposed:
        mat = mat.t()
    return mat.reshape(original_shape)


class MuonLike(torch.optim.AdamW):
    def __init__(self, params, *args, ns_steps: int = 2, ns_eps: float = 1e-7, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.ns_steps = ns_steps
        self.ns_eps = ns_eps

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.grad.ndim < 2:
                    continue
                p.grad.copy_(_orthogonalize_gradient(p.grad, steps=self.ns_steps, eps=self.ns_eps))
        return super().step(closure)
