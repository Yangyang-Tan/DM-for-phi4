"""
phi^4 lattice action and its gradient, in normalised-field (`[-1, 1]`) space.

Unified across 2D and 3D — pick via the ``spatial_dims`` argument
(default 2). Tensors are expected with a channel axis: ``(N, 1, L, ...)``.
"""

import torch


def phi4_action(phi, k, l, phi_min, phi_max, spatial_dims=2):
    """Compute phi^4 action S(φ) (with denormalization).

    Args:
        phi:           ``(N, 1, L, L)`` for 2D or ``(N, 1, L, L, L)`` for 3D,
                       in normalised ``[-1, 1]`` space.
        k, l:          hopping κ and quartic coupling λ.
        phi_min, phi_max: original-space normalisation range.
        spatial_dims:  2 or 3.

    Returns:
        ``(N,)`` tensor of per-sample action values.
    """
    p = (phi[:, 0] + 1) / 2 * (phi_max - phi_min) + phi_min
    # Forward-only neighbour sum along each spatial axis (axes 1 .. spatial_dims).
    neighbor_sum = sum(torch.roll(p, 1, dims=d) for d in range(1, spatial_dims + 1))
    sum_axes = tuple(range(1, spatial_dims + 1))
    return torch.sum(
        -2 * k * p * neighbor_sum + (1 - 2 * l) * p ** 2 + l * p ** 4,
        dim=sum_axes,
    )


def phi4_grad_S(phi, k, l, phi_min, phi_max, spatial_dims=2):
    """∂S/∂x in normalised-field space.

    Chain rule: ∂S/∂x = (∂S/∂p)(∂p/∂x) with p = (x+1)/2*(pmax-pmin)+pmin,
    so ∂p/∂x = (pmax-pmin)/2.

    Returns shape ``(N, 1, L, ...)`` matching ``phi``.
    """
    scale = (phi_max - phi_min) / 2.0
    p = (phi[:, 0] + 1) / 2 * (phi_max - phi_min) + phi_min
    nb = sum(
        torch.roll(p, +1, dims=d) + torch.roll(p, -1, dims=d)
        for d in range(1, spatial_dims + 1)
    )
    dS_dp = -2 * k * nb + 2 * (1 - 2 * l) * p + 4 * l * p ** 3
    return (dS_dp * scale).unsqueeze(1)
