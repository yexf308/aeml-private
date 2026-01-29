"""
    This module implements an arbitrary feed-forward neural network. The user can specify
    the number of neurons per layer, the activation function applied to each layer, and the
    total number of layers.

    The class has methods to compute the Jacobian matrix and Hessian tensor of the output of the
    network with respect to its inputs. It can handle batched samples of high dimensional points
    or batches of sample paths (time-series).
"""
from typing import List, Sequence, Optional, Any, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from torch import Tensor


class Activation(Protocol):
    def __call__(self, x: Tensor) -> Tensor:
        ...


class FrobeniusClipParametrization(nn.Module):
    def __init__(self, max_norm: float):
        """
        Clip the Frobenius norm of all the weights of a module.

        :param max_norm: maximum value of the Frobenius norm
        """
        super().__init__()
        self.max_norm = max_norm

    def forward(self, X):
        frob_norm = torch.norm(X, p='fro')
        if frob_norm > self.max_norm:
            X = X * (self.max_norm / frob_norm)
        return X

class FeedForwardNeuralNet(nn.Module):
    """
    A feedforward neural network class that constructs a network with specified layers
    and applies given activation functions.

    Attributes:
        layers (nn.ModuleList): A list of linear layers in the network.
        activations (callable): The activation function applied to all but the final layer.
        input_dim (int): the input dimension
        output_dim (int): the output dimension

    Methods:
        forward(x):
            Passes input tensor `x` through the network, applying the specified activation functions.

        jacobian_network(x):
            Computes the Jacobian matrix of the network's output with respect to its input.

        hessian_network(x):
            Computes the Hessian matrix of the network's output with respect to its input, for each coordinate

        jacobian_network_for_paths(x):
            Computes the Jacobian matrix of the network's output with respect to its input. Here the input
            is assumed to be a batch of sample-paths.

        hessian_network_for_paths(x):
            Computes the Hessian matrix of the network's output with respect to its input, for each coordinate.
            Here the input is assumed to be a batch of sample-paths.

        tie_weights(x):
            Tie the weights of this network to another network via transpose (but not the biases).

    """

    def __init__(self,
                 neurons: List[int],
                 activations: Sequence[Optional[Activation]],
                 spectral_normalize=False,
                 weight_normalize=False,
                 fro_normalize=False,
                 fro_max_norm=5.):
        """
        Initializes the FeedForwardNeuralNet with the given neurons and activation functions.

        Args:
            neurons (list): A list of integers where each integer represents the number of nodes in a layer.
            activations (callable): The list of activation functions to apply after each linear layer
            spectral_normalize (bool): A boolean for whether to apply spectral normalization to the layers.
        """
        super(FeedForwardNeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.activations: Sequence[Optional[Activation]] = activations
        self.input_dim = neurons[0]
        self.output_dim = neurons[-1]
        self.spectral_normalize = spectral_normalize
        self.weight_normalize = weight_normalize
        self.fro_normalize = fro_normalize
        self.fro_max_norm = fro_max_norm

        for i in range(len(neurons) - 1):
            layer = nn.Linear(neurons[i], neurons[i + 1])
            # Apply spectral normalization if specified
            if self.spectral_normalize:
                # Torch documentation say nn.utils.spectral_norm is to be deprecated
                # layer = nn.utils.spectral_norm(layer)
                layer = nn.utils.parametrizations.spectral_norm(layer)
            if self.weight_normalize:
                layer = nn.utils.parametrizations.weight_norm(layer)
            if self.fro_normalize:
                parametrize.register_parametrization(
                    layer, 'weight',
                    FrobeniusClipParametrization(self.fro_max_norm)
                )
            self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the neural network.

        Args:
            x (Tensor): The input tensor to the network.

        Returns:
            Tensor: The output tensor after passing through the network and activation functions.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            activation = self.activations[i]
            if activation is not None:
                x = activation(x)
        return x

    def jacobian_network(self, x: Tensor, method: str = "autograd") -> Tensor:
        """

        :param x:
        :param method:
        :return:
        """
        if method == "autograd":
            return self._jacobian_network_autograd(x)
        elif method == "exact":
            return self._jacobian_network_explicit(x)
        raise ValueError(f"Unsupported Jacobian method: {method}")

    def _jacobian_network_autograd(self, x: Tensor) -> Tensor:
        """
        Computes the Jacobian matrix of the network's output with respect to its input.

        Args:
            x (Tensor): The input tensor to the network. The tensor should have `requires_grad` enabled.

        Returns:
            Tensor: A tensor representing the Jacobian matrix of the network's output with respect to the input.
        """
        x = x.requires_grad_(True)
        y = self.forward(x)
        jacobian = []
        for i in range(self.output_dim):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1
            jacob = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True)[0]
            jacobian.append(jacob)
        jacobian = torch.stack(jacobian, dim=1)
        return jacobian

    def _jacobian_network_explicit(self, x: Tensor):
        """
        Computes the Jacobian matrix of the network's output with respect to its input explicitly.

        Args:
            x (Tensor): The input tensor to the network of shape (batch_size, input_dim).

        Returns:
            Tensor: A tensor representing the Jacobian matrix of the network's output
                          with respect to the input, of shape (batch_size, output_dim, input_dim).
        """
        # Ensure input is batched
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.size(0)

        # Get weights of each layer
        weights: List[Tensor] = [cast(Tensor, layer.weight) for layer in self.layers]

        # Initialize Jacobian with the first layer's weight
        jacobian = weights[0].repeat(batch_size, 1, 1)

        # Forward pass to store intermediate values
        y = x
        z_values = []
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            z = layer(y)
            z_values.append(z.detach().requires_grad_())

            if activation is not None:
                y = activation(z_values[-1])
            else:
                y = z_values[-1]

        # Backward pass to compute Jacobian
        for i in range(1, len(self.layers)):
            z = z_values[i - 1]

            activation = self.activations[i - 1]
            if activation is not None:
                # Use autograd to compute the derivative of the activation function
                a = activation(z)
                diag = torch.autograd.grad(a.sum(), z, create_graph=False)[0]
                diag_term = torch.diag_embed(diag.view(batch_size, -1))
            else:
                diag_term = torch.eye(z.size(1)).unsqueeze(0).repeat(batch_size, 1, 1)

            next_term = torch.bmm(weights[i].repeat(batch_size, 1, 1), diag_term)
            jacobian = torch.bmm(next_term, jacobian)

        # Apply final activation if present
        activation = self.activations[-1]
        if activation is not None:
            z = z_values[-1]
            a = activation(z)
            diag = torch.autograd.grad(a.sum(), z, create_graph=False)[0]
            diag_term = torch.diag_embed(diag.view(batch_size, -1))
            jacobian = torch.bmm(diag_term, jacobian)

        return jacobian

    def hessian_network(self, x: Tensor):
        """
        Computes the Hessian matrix of the network's output with respect to its input.

        Args:
            x (Tensor): The input tensor to the network of shape (n, d),
                              where n is the batch size and d is the input dimension.

        Returns:
            Tensor: A tensor representing the Hessian matrix of the network's output
                          with respect to the input, of shape (n, self.output_dim, d, d).
        """
        n, d = x.shape
        x.requires_grad_(True)
        y = self.forward(x)

        hessians = []
        for i in range(self.output_dim):
            # Compute first-order gradients
            first_grads = torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0]

            # Compute second-order gradients (Hessian)
            hessian_rows = []
            for j in range(d):
                hessian_row = torch.autograd.grad(first_grads[:, j].sum(), x, retain_graph=True)[0]
                hessian_rows.append(hessian_row)

            hessian = torch.stack(hessian_rows, dim=1)
            hessians.append(hessian)

        # Stack Hessians for each output dimension
        hessians = torch.stack(hessians, dim=1)

        return hessians

    def jacobian_network_for_paths(self, x: Tensor):
        """
        Computes the Jacobian matrix of the network's output with respect to its input.

        Args:
            x (Tensor): The input tensor to the network of shape (num_paths, n+1, d). The tensor should have
            `requires_grad` enabled.

        Returns:
            Tensor: A tensor representing the Jacobian matrix of the network's output with respect to the input.
        """
        num_paths, n, d = x.size()
        x.requires_grad_(True)
        jacobian = torch.zeros(num_paths, n, d, self.output_dim)
        for i in range(num_paths):
            for j in range(n):
                x_ij = x[i, j, :].unsqueeze(0)  # shape (1, d)
                output = self.forward(x_ij)  # shape (1, D)
                for k in range(self.output_dim):
                    grad_outputs = torch.zeros_like(output)
                    grad_outputs[0, k] = 1
                    gradients = torch.autograd.grad(outputs=output, inputs=x_ij, grad_outputs=grad_outputs,
                                                    create_graph=True)[0]  # shape (1, d)
                    jacobian[i, j, :, k] = gradients.squeeze()
        return jacobian.transpose(2, 3)

    def hessian_network_for_paths(self, x: Tensor):
        """
        Computes the Hessian matrix of the network's output with respect to its input.

        Args:
            x (Tensor): The input tensor to the network of shape (num_paths, n+1, d). The tensor should have
            `requires_grad` enabled.

        Returns:
            Tensor: A tensor representing the Hessian matrix of the network's output with respect to the input.
        """
        num_paths, n, d = x.shape
        x.requires_grad_(True)
        outputs = self.forward(x)

        hessians = torch.zeros(num_paths, n, self.output_dim, d, d)

        for i in range(num_paths):
            for j in range(n):
                for k in range(self.output_dim):
                    grad_outputs = torch.zeros_like(outputs)
                    grad_outputs[i, j, k] = 1
                    grads = \
                        torch.autograd.grad(outputs, x, grad_outputs=grad_outputs, create_graph=True,
                                            allow_unused=True)[0]
                    if grads is None:
                        continue
                    for r in range(d):
                        hessian_row = torch.autograd.grad(grads[i, j, r], x, retain_graph=True, allow_unused=True)[0]
                        if hessian_row is not None:
                            hessians[i, j, k, r, :] = hessian_row[i, j, :]

        return hessians

    def tie_weights(self, other: "FeedForwardNeuralNet"):
        """
        Tie the weights of this network to the transpose of another one.
        """
        for layer_self, layer_other in zip(self.layers, reversed(other.layers)):
            if isinstance(layer_self, nn.Linear) and isinstance(layer_other, nn.Linear):
                with torch.no_grad():
                    layer_self.weight.copy_(layer_other.weight.t())

    def frobenius_inner_product_jvp(self, latent_covariance: Tensor,
                                    x: Tensor) -> Tensor:
        """
        Computes the ambient quadratic variation drift using a JVP-based method.
        For each sample n and each output component i, it computes:
            q^i(x_n) = sum_{j=1}^d  v_j^T * (Hessian of f^i at x_n) * v_j,
        where the vectors {v_j} are the columns of the Cholesky factor L of latent_covariance,
        i.e., latent_covariance = L L^T.

        Args:
            latent_covariance (Tensor): Tensor of shape (n, d, d) for each sample’s covariance matrix.
            net (FeedForwardNeuralNet): The neural network f.
            x (Tensor): Input tensor of shape (n, d) with requires_grad enabled.

        Returns:
            Tensor: A tensor of shape (n, output_dim) with the computed q values.
        """
        n, d = x.shape
        # Compute network output to determine output dimension.
        f_val = self.forward(x)
        if f_val.dim() == 1:
            r = 1
            f_val = f_val.unsqueeze(1)
        else:
            r = f_val.shape[1]

        q = torch.zeros(n, r, device=x.device, dtype=x.dtype)

        # Loop over each sample in the batch.
        for idx in range(n):
            # Compute the Cholesky factor of latent_covariance for this sample.
            # Add a small term to the diagonal for numerical stability.
            L = torch.linalg.cholesky(latent_covariance[idx] + 1e-6 * torch.eye(d, device=x.device, dtype=x.dtype))
            # Loop over the columns of L (each column is a direction vector v).
            for j in range(d):
                v = L[:, j]  # shape (d,)
                # For each output component i, compute the second directional derivative.
                for i in range(r):
                    # Define a scalar function that returns the i-th output for a single sample.
                    def scalar_f(x_single):
                        # x_single is assumed to be of shape (1, d)
                        out = self.forward(x_single)
                        if out.dim() == 1:
                            return out[i]
                        else:
                            return out[0, i]

                    # Extract the single sample.
                    x_single = x[idx:idx + 1]  # shape (1, d)
                    # Compute the first derivative (gradient) of scalar_f at x_single.
                    grad_f = torch.autograd.grad(scalar_f(x_single), x_single, create_graph=True)[0]  # shape (1, d)
                    # Compute the directional derivative: grad_f dot v.
                    directional_deriv = torch.sum(grad_f * v.unsqueeze(0))
                    # Compute the second directional derivative: derivative of (grad_f dot v) along v.
                    second_deriv = torch.autograd.grad(directional_deriv, x_single, retain_graph=True)[0]
                    # Dot with v to get v^T Hessian f^i(x) v.
                    hvp = torch.sum(second_deriv * v.unsqueeze(0))
                    q[idx, i] += hvp
        return q
