import torch


class SpectralNorm(object):
    def __init__(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                "got n_power_iterations={}".format(n_power_iterations)
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_sigma(self, module):
        with torch.no_grad():
            weight = getattr(module, self.name)
            weight_mat = self.reshape_weight_to_matrix(weight)

            u = getattr(module, self.name + "_u")
            v = getattr(module, self.name + "_v")
            for _ in range(self.n_power_iterations):
                v = torch.nn.functional.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps)
                u = torch.nn.functional.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps)
            setattr(module, self.name + "_u", u)
            setattr(module, self.name + "_v", v)

            sigma = torch.dot(u, torch.mv(weight_mat, v))
            setattr(module, "spectral_norm", sigma)

    def remove(self, module):
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, "spectral_norm")

    def __call__(self, module, inputs):
        self.compute_sigma(module)

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps, normalize=True):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on " "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            u = torch.nn.functional.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = torch.nn.functional.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer("spectral_norm", torch.tensor(-1, device=next(module.parameters()).device))

        module.register_forward_pre_hook(fn)
        return fn


def track_spectral_norm(module, name="weight", n_power_iterations=1, eps=1e-12, dim=None):
    r"""Tracks the spectral norm of a module's weight parameter
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``
    Returns:
        The original module with the spectral norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])
    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name="weight"):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))

    return module
