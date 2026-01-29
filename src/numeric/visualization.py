from typing import Optional

import numpy as np
import torch


def lift_sample_paths(decoder, latent_ensemble: np.ndarray) -> np.ndarray:
    """
    Lift latent paths to the ambient space using a decoder.
    """
    lifted_ensemble = np.array([
        decoder(torch.tensor(path, dtype=torch.float32)).detach().numpy()
        for path in latent_ensemble
    ])
    return lifted_ensemble


def plot_surface(decoder,
                 a: float,
                 b: float,
                 grid_size: int,
                 ax=None,
                 title: Optional[str] = None,
                 dim: int = 3,
                 device: str = "cpu") -> None:
    """
    Plot the surface produced by a decoder network.
    """
    if dim == 3:
        ux = np.linspace(a, b, grid_size)
        vy = np.linspace(a, b, grid_size)
        u, v = np.meshgrid(ux, vy, indexing="ij")
        x1 = np.zeros((grid_size, grid_size))
        x2 = np.zeros((grid_size, grid_size))
        x3 = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                x0 = np.column_stack([u[i, j], v[i, j]])
                x0 = torch.tensor(x0, dtype=torch.float32, device=device)
                xx = decoder(x0).cpu().detach().numpy()
                x1[i, j] = xx[0, 0]
                x2[i, j] = xx[0, 1]
                x3[i, j] = xx[0, 2]
        if ax is not None:
            ax.plot_surface(x1, x2, x3, alpha=0.5, cmap="magma")
            if title is not None:
                ax.set_title(title)
            else:
                ax.set_title("NN manifold")
        else:
            raise ValueError("'ax' cannot be None")
    elif dim == 2:
        u = np.linspace(a, b, grid_size)
        x1 = np.zeros((grid_size, grid_size))
        x2 = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            x0 = np.column_stack([u[i]])
            x0 = torch.tensor(x0, dtype=torch.float32, device=device)
            xx = decoder(x0).detach().numpy()
            x1[i] = xx[0, 0]
            x2[i] = xx[0, 1]
        if ax is not None:
            ax.plot(x1, x2, alpha=0.9)
            if title is not None:
                ax.set_title(title)
            else:
                ax.set_title("NN manifold")
    return None
