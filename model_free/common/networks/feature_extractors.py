"""Feature extractor modules for vector and image observations.

This module contains MLP/CNN encoders and noisy-layer variants used by
policy/value/Q networks. The extractors expose stable output feature sizes so
heads can be assembled without duplicating encoder logic across algorithms.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Type

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from rllib.model_free.common.utils.network_utils import _validate_hidden_sizes, _make_weights_init


# =============================================================================
# MLP Feature Extractors
# =============================================================================

class MLPFeaturesExtractor(nn.Module):
    """
    Standard MLP feature extractor (shared trunk / encoder).

    This module is intentionally minimal and is meant to be reused as a
    *shared trunk* for multiple heads (actor/critic, value/Q, etc.).

    Parameters
    ----------
    input_dim : int
        Input dimensionality (e.g., observation/state dimension).
    hidden_sizes : list[int]
        Hidden layer widths. The output feature dimension is ``hidden_sizes[-1]``.
        Must contain at least one element.
    activation_fn : type[nn.Module], optional
        Activation module class inserted after each ``nn.Linear`` layer
        (default: ``nn.ReLU``).

    Attributes
    ----------
    net : nn.Sequential
        Sequential stack: (Linear -> Activation) repeated.
    out_dim : int
        Output feature dimensionality (= ``hidden_sizes[-1]``).

    Notes
    -----
    - Architecture: (Linear → Activation) × N
    - No normalization/residual connections are included by design.
    - This is a pure feature extractor; distribution/value heads are built elsewhere.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        activation_fn: type[nn.Module] = nn.ReLU,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        input_dim : Any
            Argument ``input_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        hidden_sizes : Any
            Argument ``hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        super().__init__()

        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one layer size.")

        layers: list[nn.Module] = []
        prev_dim = int(input_dim)

        for h in hidden_sizes:
            h = int(h)
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_fn())
            prev_dim = h

        self.net = nn.Sequential(*layers)
        self.out_dim = int(hidden_sizes[-1])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(B, out_dim)``.
        """
        return self.net(x)


# =============================================================================
# CNN Feature Extractors
# =============================================================================

class ConvNet(nn.Module):
    """
    Configurable convolutional feature extractor.

    This module builds a stack of Conv2d layers followed by activation
    functions and optionally flattens the output into a 2D tensor suitable
    for fully connected heads (e.g., actor, critic, value, or Q networks).

    The output dimensionality (`out_dim`) is automatically inferred using
    a dummy forward pass at initialization.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of input tensor in (C, H, W) format.

    conv_channels : tuple[int, ...], optional
        Number of output channels for each convolutional layer.
        Default is (32, 64, 64).

    kernel_sizes : tuple[int, ...], optional
        Kernel size for each convolutional layer.
        Must have the same length as conv_channels.
        Default is (8, 4, 3).

    strides : tuple[int, ...], optional
        Stride for each convolutional layer.
        Must have the same length as conv_channels.
        Default is (4, 2, 1).

    activation_fn : Type[nn.Module], optional
        Activation function class applied after each convolution.
        Default is nn.ReLU.

    flatten : bool, optional
        Whether to flatten output into shape (B, out_dim).
        If False, output retains shape (B, C, H, W).
        Default is True.

    init_type : str, optional
        Weight initialization method passed to `_make_weights_init`.
        Common values include:
        - "orthogonal"
        - "xavier"
        Default is "orthogonal".

    gain : float, optional
        Gain parameter for weight initialization.
        Default is 1.0.

    bias : float, optional
        Initial bias value for all layers.
        Default is 0.0.

    Attributes
    ----------
    cnn : nn.Sequential
        Sequential convolutional network.

    flatten : bool
        Whether flattening is applied.

    out_dim : int
        Output feature dimensionality after flattening.

    Raises
    ------
    ValueError
        If input_shape is invalid or parameter lengths mismatch.

    Notes
    -----
    Output dimensionality is computed automatically using a dummy forward pass.

    Typical usage in RL:

    >>> conv = ConvNet(input_shape=(4, 84, 84))
    >>> features = conv(obs)
    >>> features.shape
    torch.Size([batch_size, out_dim])

    This design is compatible with:

    - DQN
    - SAC
    - PPO
    - Actor-Critic architectures
    - Multi-head networks
    """

    def __init__(
        self,
        *,
        input_shape: Tuple[int, int, int],
        conv_channels: Tuple[int, ...] = (32, 64, 64),
        kernel_sizes: Tuple[int, ...] = (8, 4, 3),
        strides: Tuple[int, ...] = (4, 2, 1),
        activation_fn: Type[nn.Module] = nn.ReLU,
        flatten: bool = True,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        """
        Initialize ConvNet.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            Input shape in (C, H, W) format.

        conv_channels : tuple[int, ...]
            Output channels per convolution layer.

        kernel_sizes : tuple[int, ...]
            Kernel sizes per convolution layer.

        strides : tuple[int, ...]
            Strides per convolution layer.

        activation_fn : Type[nn.Module]
            Activation function class.

        flatten : bool
            Whether to flatten output.

        init_type : str
            Weight initialization type.

        gain : float
            Initialization gain.

        bias : float
            Bias initialization value.
        """
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (C,H,W), got {input_shape}")

        if not (len(conv_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(
                "conv_channels, kernel_sizes, strides must have same length"
            )

        c, h, w = map(int, input_shape)

        layers = []
        in_ch = c

        for out_ch, k, s in zip(conv_channels, kernel_sizes, strides):

            layers.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=int(out_ch),
                    kernel_size=int(k),
                    stride=int(s),
                )
            )

            layers.append(activation_fn())

            in_ch = int(out_ch)

        self.cnn = nn.Sequential(*layers)

        self.flatten = flatten

        # Initialize weights
        init_fn = _make_weights_init(init_type, gain, bias)
        self.apply(init_fn)

        # Infer output dimension
        with th.no_grad():

            dummy = th.zeros(1, c, h, w)

            out = self.cnn(dummy)

            if flatten:
                self.out_dim = int(out.view(1, -1).shape[1])
            else:
                self.out_dim = int(out.numel() // out.shape[0])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape:

            - (B, out_dim) if flatten=True
            - (B, C_out, H_out, W_out) otherwise
        """
        x = self.cnn(x)

        if self.flatten:
            x = x.view(x.size(0), -1)

        return x


class CNNFeaturesExtractor(nn.Module):
    """
    CNN-based feature extractor module.

    This module wraps the `ConvNet` backbone and exposes a standardized
    interface for extracting feature vectors from image observations.

    It is typically used as a shared encoder (trunk) in reinforcement
    learning architectures such as Actor-Critic, DQN, SAC, and PPO.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of input tensor in (C, H, W) format.

    conv_channels : tuple[int, ...], optional
        Number of output channels for each convolutional layer.
        Default is (32, 64, 64).

    kernel_sizes : tuple[int, ...], optional
        Kernel sizes for each convolutional layer.
        Must have same length as conv_channels.
        Default is (8, 4, 3).

    strides : tuple[int, ...], optional
        Strides for each convolutional layer.
        Must have same length as conv_channels.
        Default is (4, 2, 1).

    activation_fn : Type[nn.Module], optional
        Activation function class applied after each convolution.
        Default is nn.ReLU.

    flatten : bool, optional
        Whether to flatten output into shape (B, out_dim).
        Default is True.

    init_type : str, optional
        Weight initialization method passed to ConvNet.
        Supported values typically include:
        - "orthogonal"
        - "xavier"
        Default is "orthogonal".

    gain : float, optional
        Gain parameter for weight initialization.
        Default is 1.0.

    bias : float, optional
        Initial bias value.
        Default is 0.0.

    Attributes
    ----------
    net : ConvNet
        Underlying convolutional feature extractor.

    out_dim : int
        Output feature dimensionality.

    Notes
    -----
    This module provides a standardized interface so downstream
    networks (e.g., policy head, value head, Q-network) can access
    feature dimension via `out_dim`.

    Typical RL pipeline:

    observation -> CNNFeaturesExtractor -> feature vector -> head network

    Examples
    --------
    >>> extractor = CNNFeaturesExtractor(input_shape=(4, 84, 84))
    >>> obs = torch.randn(32, 4, 84, 84)
    >>> features = extractor(obs)
    >>> features.shape
    torch.Size([32, extractor.out_dim])

    Compatible with:

    - DQN
    - SAC
    - PPO
    - A2C / A3C
    - Actor-Critic architectures
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        *,
        conv_channels: Tuple[int, ...] = (32, 64, 64),
        kernel_sizes: Tuple[int, ...] = (8, 4, 3),
        strides: Tuple[int, ...] = (4, 2, 1),
        activation_fn: Type[nn.Module] = nn.ReLU,
        flatten: bool = True,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        """
        Initialize CNNFeaturesExtractor.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            Input shape in (C, H, W) format.

        conv_channels : tuple[int, ...]
            Output channels per convolution layer.

        kernel_sizes : tuple[int, ...]
            Kernel sizes per convolution layer.

        strides : tuple[int, ...]
            Strides per convolution layer.

        activation_fn : Type[nn.Module]
            Activation function class.

        flatten : bool
            Whether to flatten output.

        init_type : str
            Weight initialization method.

        gain : float
            Initialization gain.

        bias : float
            Bias initialization value.
        """
        super().__init__()

        self.net = ConvNet(
            input_shape=input_shape,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            activation_fn=activation_fn,
            flatten=flatten,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        self.out_dim = self.net.out_dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, out_dim)

        Notes
        -----
        This method simply delegates forward computation to the underlying
        ConvNet backbone.
        """
        return self.net(x)


# =============================================================================
# Noisy Networks (Rainbow / Exploration-friendly)
# =============================================================================

class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer (Fortunato et al.).

    This layer replaces deterministic Linear weights with:
        W = W_mu + W_sigma ⊙ eps_W
        b = b_mu + b_sigma ⊙ eps_b
    where eps is sampled from a factorized noise distribution.

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    std_init : float, optional
        Initial value for sigma parameters (default: 0.5).

    Attributes
    ----------
    weight_mu : nn.Parameter
        Mean of weights, shape ``(out_features, in_features)``.
    weight_sigma : nn.Parameter
        Std (scale) of weights, same shape.
    weight_epsilon : torch.Tensor (buffer)
        Sampled noise for weights, same shape.
    bias_mu : nn.Parameter
        Mean of bias, shape ``(out_features,)``.
    bias_sigma : nn.Parameter
        Std (scale) of bias, shape ``(out_features,)``.
    bias_epsilon : torch.Tensor (buffer)
        Sampled noise for bias, shape ``(out_features,)``.

    Notes
    -----
    - Call `reset_noise()` to resample epsilons.
    - `reset_noise()` mutates epsilon buffers in-place; `forward()` clones epsilons
      to avoid autograd version-counter errors if noise is reset between forwards.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.5,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        in_features : Any
            Argument ``in_features`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        out_features : Any
            Argument ``out_features`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        std_init : Any
            Argument ``std_init`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = float(std_init)

        self.weight_mu = nn.Parameter(th.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(th.empty(self.out_features, self.in_features))
        self.register_buffer("weight_epsilon", th.empty(self.out_features, self.in_features))

        self.bias_mu = nn.Parameter(th.empty(self.out_features))
        self.bias_sigma = nn.Parameter(th.empty(self.out_features))
        self.register_buffer("bias_epsilon", th.empty(self.out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """
        Initialize mu and sigma parameters.

        Notes
        -----
        - mu is uniform in [-1/sqrt(in), 1/sqrt(in)]
        - sigma is constant scaled by std_init
        """
        mu_range = 1.0 / math.sqrt(self.in_features)
        with th.no_grad():
            self.weight_mu.uniform_(-mu_range, mu_range)
            self.weight_sigma.fill_(self.std_init / math.sqrt(self.in_features))

            self.bias_mu.uniform_(-mu_range, mu_range)
            self.bias_sigma.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: th.device, dtype: th.dtype) -> th.Tensor:
        """
        Sample noise vector using factorized form: f(x)=sign(x)*sqrt(|x|).

        Returns
        -------
        torch.Tensor
            Noise vector of shape ``(size,)``.
        """
        x = th.randn(size, device=device, dtype=dtype)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """
        Resample factorized noise for weights and biases.

        Notes
        -----
        - Epsilon buffers are updated in-place.
        - Safe with autograd as long as forward does not keep references to the same
          storage; forward clones epsilons to avoid version-counter issues.
        """
        device = self.weight_epsilon.device
        dtype = self.weight_epsilon.dtype

        eps_in = self._scale_noise(self.in_features, device=device, dtype=dtype)
        eps_out = self._scale_noise(self.out_features, device=device, dtype=dtype)

        with th.no_grad():
            self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))  # (out, in)
            self.bias_epsilon.copy_(eps_out)  # (out,)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass with noisy parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, out_features)``.

        Notes
        -----
        - `clone()` is critical: `detach()` alone shares storage and may still trigger
          "modified by an inplace operation" errors if `reset_noise()` happens later.
        """
        w_eps = self.weight_epsilon.detach().clone()
        b_eps = self.bias_epsilon.detach().clone()

        w = self.weight_mu + self.weight_sigma * w_eps
        b = self.bias_mu + self.bias_sigma * b_eps
        return F.linear(x, w, b)


class NoisyMLPFeaturesExtractor(nn.Module):
    """
    MLP feature extractor with NoisyLinear layers.

    Architecture
    ------------
    - First layer: deterministic `nn.Linear` (often stabilizes early learning)
    - Remaining layers: `NoisyLinear` (exploration via parameter noise)

    Parameters
    ----------
    input_dim : int
        Input dimensionality.
    hidden_sizes : tuple[int, ...], optional
        Hidden layer sizes (default: (64, 64)).
    activation_fn : type[nn.Module], optional
        Activation module class (default: ``nn.ReLU``).
    init_type : str, optional
        Initializer name for deterministic layers (default: "orthogonal").
    gain : float, optional
        Init gain (default: 1.0).
    bias : float, optional
        Bias init constant (default: 0.0).
    noisy_std_init : float, optional
        Initial sigma for `NoisyLinear` (default: 0.5).

    Attributes
    ----------
    input_layer : nn.Linear
        Deterministic first layer.
    hidden_layers : nn.ModuleList
        List of `NoisyLinear` layers.
    activation : nn.Module
        Activation instance.
    out_dim : int
        Output feature dimensionality.

    Notes
    -----
    Call `reset_noise()` to resample noise in all NoisyLinear layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        noisy_std_init: float = 0.5,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        input_dim : Any
            Argument ``input_dim`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        hidden_sizes : Any
            Argument ``hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_type : Any
            Argument ``init_type`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        gain : Any
            Argument ``gain`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        bias : Any
            Argument ``bias`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        noisy_std_init : Any
            Argument ``noisy_std_init`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        super().__init__()

        hs = _validate_hidden_sizes(hidden_sizes)
        self.activation = activation_fn()

        self.input_layer = nn.Linear(int(input_dim), int(hs[0]))
        self.hidden_layers = nn.ModuleList(
            [
                NoisyLinear(int(hs[i]), int(hs[i + 1]), std_init=noisy_std_init)
                for i in range(len(hs) - 1)
            ]
        )

        init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
        self.input_layer.apply(init_fn)

        self.out_dim = int(hs[-1])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, input_dim)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(B, out_dim)``.
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self) -> None:
        """
        Resample noise in all NoisyLinear layers.
        """
        for layer in self.hidden_layers:
            layer.reset_noise()


class NoisyCNNFeaturesExtractor(nn.Module):
    """
    CNN feature extractor with NoisyLinear MLP head.

    This module combines:
    - A convolutional trunk (`ConvNet`) for image observations, and
    - A fully-connected head where some layers are `NoisyLinear` to encourage
      exploration via parameter noise (Fortunato et al.).

    Architecture
    ------------
    observation (B, C, H, W)
        -> ConvNet (flatten=True) => (B, cnn_out_dim)
        -> [Linear (deterministic) + activation]   (optional stabilization)
        -> [NoisyLinear + activation] x (N-1)
        -> features (B, out_dim)

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Input shape in (C, H, W) format.

    conv_channels : tuple[int, ...], optional
        Output channels per convolution layer. Default is (32, 64, 64).

    kernel_sizes : tuple[int, ...], optional
        Kernel sizes per convolution layer. Default is (8, 4, 3).

    strides : tuple[int, ...], optional
        Strides per convolution layer. Default is (4, 2, 1).

    head_hidden_sizes : tuple[int, ...], optional
        Hidden sizes of the MLP head. Must be non-empty.
        The final feature dimension is `head_hidden_sizes[-1]`.
        Default is (256, 256).

    activation_fn : Type[nn.Module], optional
        Activation function class used after conv and FC layers.
        Default is nn.ReLU.

    init_type : str, optional
        Initialization type for deterministic layers (ConvNet conv + first Linear).
        Passed to `_make_weights_init`. Default is "orthogonal".

    gain : float, optional
        Gain for initialization. Default is 1.0.

    bias : float, optional
        Bias init constant. Default is 0.0.

    noisy_std_init : float, optional
        Initial sigma for NoisyLinear layers. Default is 0.5.

    deterministic_first_layer : bool, optional
        If True, the first FC layer is deterministic `nn.Linear` (often stabilizes
        early learning), and the remaining FC layers are `NoisyLinear`.
        If False, all FC layers are `NoisyLinear`.
        Default is True.

    Attributes
    ----------
    cnn : ConvNet
        Convolutional trunk (always `flatten=True`).

    out_dim : int
        Output feature dimensionality.

    Notes
    -----
    - Call `reset_noise()` at the desired frequency (e.g., each environment step,
      each forward, or each rollout fragment) depending on your algorithm.
    - This module only provides features; policy/value/Q heads should be built
      separately.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        *,
        conv_channels: Tuple[int, ...] = (32, 64, 64),
        kernel_sizes: Tuple[int, ...] = (8, 4, 3),
        strides: Tuple[int, ...] = (4, 2, 1),
        head_hidden_sizes: Tuple[int, ...] = (256, 256),
        activation_fn: Type[nn.Module] = nn.ReLU,
        init_type: str = "orthogonal",
        gain: float = 1.0,
        bias: float = 0.0,
        noisy_std_init: float = 0.5,
        deterministic_first_layer: bool = True,
    ) -> None:
        """Initialize this module.

        Parameters
        ----------
        input_shape : Any
            Argument ``input_shape`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        conv_channels : Any
            Argument ``conv_channels`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        kernel_sizes : Any
            Argument ``kernel_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        strides : Any
            Argument ``strides`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        head_hidden_sizes : Any
            Argument ``head_hidden_sizes`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        activation_fn : Any
            Argument ``activation_fn`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        init_type : Any
            Argument ``init_type`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        gain : Any
            Argument ``gain`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        bias : Any
            Argument ``bias`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        noisy_std_init : Any
            Argument ``noisy_std_init`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.
        deterministic_first_layer : Any
            Argument ``deterministic_first_layer`` used to configure the module or provide runtime input; expected shape and semantics are governed by this class's API.

        Returns
        -------
        Any
            Returns tensors or scalars consistent with this method's documented shape, dtype, and batching conventions.
        """
        super().__init__()

        hs = _validate_hidden_sizes(head_hidden_sizes)
        self.activation = activation_fn()

        # CNN trunk: always flatten to (B, cnn_out_dim)
        self.cnn = ConvNet(
            input_shape=input_shape,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            activation_fn=activation_fn,
            flatten=True,
            init_type=init_type,
            gain=gain,
            bias=bias,
        )

        cnn_out_dim = int(self.cnn.out_dim)

        # FC head
        self.deterministic_first_layer = bool(deterministic_first_layer)

        if self.deterministic_first_layer:
            # First layer deterministic
            self.fc0 = nn.Linear(cnn_out_dim, int(hs[0]))
            init_fn = _make_weights_init(init_type=init_type, gain=gain, bias=bias)
            self.fc0.apply(init_fn)

            # Remaining layers noisy
            self.noisy_layers = nn.ModuleList(
                [
                    NoisyLinear(int(hs[i]), int(hs[i + 1]), std_init=noisy_std_init)
                    for i in range(len(hs) - 1)
                ]
            )
        else:
            # All layers noisy
            self.fc0 = None
            self.noisy_layers = nn.ModuleList(
                [
                    NoisyLinear(
                        cnn_out_dim if i == 0 else int(hs[i - 1]),
                        int(hs[i]),
                        std_init=noisy_std_init,
                    )
                    for i in range(len(hs))
                ]
            )

        self.out_dim = int(hs[-1])

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Image observation tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, out_dim).
        """
        x = self.cnn(x)  # (B, cnn_out_dim)

        if self.deterministic_first_layer:
            x = self.activation(self.fc0(x))
            for layer in self.noisy_layers:
                x = self.activation(layer(x))
        else:
            for layer in self.noisy_layers:
                x = self.activation(layer(x))

        return x

    def reset_noise(self) -> None:
        """
        Resample noise in all NoisyLinear layers.
        """
        for layer in self.noisy_layers:
            layer.reset_noise()


# =============================================================================
# Builder
# =============================================================================
def build_feature_extractor(
    *,
    obs_dim: int,
    obs_shape: Optional[Tuple[int, ...]] = None,
    feature_extractor_cls: Optional[Type[nn.Module]] = None,
    feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
    input_dim_override: Optional[int] = None,
) -> tuple[nn.Module | None, int | None]:
    """Build a feature extractor and infer its output dimension.

    Parameters
    ----------
    obs_dim : int
        Observation dimension used for MLP-like extractors.
    obs_shape : tuple[int, ...], optional
        Observation shape used for CNN-like extractors.
    feature_extractor_cls : type[nn.Module], optional
        Extractor class to instantiate. If ``None``, no extractor is built.
    feature_extractor_kwargs : dict[str, Any], optional
        Extra keyword arguments passed to the extractor constructor.
    input_dim_override : int, optional
        If provided, overrides ``obs_dim`` when wiring MLP input dimensions.

    Returns
    -------
    tuple[nn.Module | None, int | None]
        ``(extractor, feature_dim)``. If no extractor class is provided,
        returns ``(None, None)``.

    Raises
    ------
    ValueError
        If a CNN extractor is requested without ``obs_shape`` or if the
        extractor does not expose ``out_dim``/``feature_dim``.
    TypeError
        If ``feature_extractor_cls`` is not a class.
    """
    if feature_extractor_cls is None:
        return None, None

    if not isinstance(feature_extractor_cls, type):
        raise TypeError(
            "feature_extractor_cls must be a class derived from nn.Module, "
            f"got {type(feature_extractor_cls)}"
        )

    kwargs = dict(feature_extractor_kwargs or {})

    effective_obs_dim = int(input_dim_override) if input_dim_override is not None else int(obs_dim)

    if issubclass(feature_extractor_cls, (CNNFeaturesExtractor, NoisyCNNFeaturesExtractor)):
        if obs_shape is None:
            raise ValueError("obs_shape must be provided for CNN feature extractors.")
        input_shape = tuple(int(x) for x in obs_shape)
        extractor = feature_extractor_cls(input_shape=input_shape, **kwargs)
    elif issubclass(feature_extractor_cls, (MLPFeaturesExtractor, NoisyMLPFeaturesExtractor)):
        extractor = feature_extractor_cls(input_dim=effective_obs_dim, **kwargs)
    else:
        if obs_shape is not None and "input_shape" not in kwargs:
            kwargs["input_shape"] = tuple(int(x) for x in obs_shape)
        if "input_dim" not in kwargs and "obs_dim" not in kwargs:
            kwargs["input_dim"] = effective_obs_dim
        extractor = feature_extractor_cls(**kwargs)

    feature_dim = getattr(extractor, "out_dim", None)
    if feature_dim is None:
        feature_dim = getattr(extractor, "feature_dim", None)
    if feature_dim is None:
        raise ValueError("feature_extractor must provide out_dim or feature_dim.")

    return extractor, int(feature_dim)
