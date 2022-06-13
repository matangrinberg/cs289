
import numpy as np
import sys
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Tuple, Union
from typing_extensions import Literal
from neural_networks.utils import convolution


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(self, param_name: str, X: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(self, n_out: int, activation: str, weight_init="xavier_uniform") -> None:
        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]
        # initialize weights / biases
        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))

        # initialize dicts
        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": 0.0})  # cache for backprop
        dW = np.zeros((self.n_in, self.n_out))
        db = np.zeros((1, self.n_out))
        dX = np.zeros((1, self.n_in))  # THIS IS PROBABLY WRONG. WHAT SHOULD FIRST DIMENSION BE?
        self.gradients = OrderedDict({"W": dW, "b": db, "X": dX})  # parameter gradients initialized to zero

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)
        W, b = self.parameters["W"], self.parameters["b"]

        # perform an affine transformation and activation
        Z = X @ W + b
        out = self.activation.forward(Z)
        
        # store information necessary for backprop in `self.cache`
        self.cache['Z'] = Z
        self.cache['X'] = X

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        
        # unpack the cache
        Z, W, X = self.cache["Z"], self.parameters["W"], self.cache["X"]
        
        # compute the gradients of the loss w.r.t. all parameters as well as the input of the layer
        dLdZ = self.activation.backward(Z, dLdY)
        dLdW = np.transpose(X) @ dLdZ
        dLdb = np.transpose(np.ones(X.shape[0])) @ dLdZ
        dLdX = dLdZ @ np.transpose(W)

        # store the gradients in `self.gradients`
        self.gradients["W"] = dLdW
        self.gradients["b"] = dLdb
        self.gradients["X"] = dLdX

        return dLdX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(self, n_out: int, kernel_shape: Tuple[int, int], activation: str,
                 stride: int = 1, pad: str = "same", weight_init: str = "xavier_uniform") -> None:
        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": [], "X": []})
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        X_padded, p = convolution.pad2d(X, self.pad, kernel_shape, stride=self.stride)
        out_rows = (in_rows - kernel_height + 2 * self.pad[0]) // self.stride + 1
        out_cols = (in_cols - kernel_width + 2 * self.pad[1]) // self.stride + 1
        out_shape = (out_rows, out_cols)

        Z = np.zeros((n_examples, out_rows, out_cols, out_channels))

        # implement a convolutional forward pass
        for d1 in range(out_rows):
            d1_step = d1 * self.stride
            for d2 in range(out_cols):
                d2_step = d2 * self.stride
                x = X_padded[:, d1_step:d1_step + kernel_height, d2_step:d2_step + kernel_width, :]
                Z[:, d1, d2, :] = np.einsum('ijkl, jklm', x, W) + b

        out = self.activation.forward(Z)

        # cache any values required for backprop
        self.cache["Z"] = Z
        self.cache["X"] = X
        self.cache["X_p"] = X_padded
        self.cache["p"] = p
        self.cache["out_shape"] = out_shape

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """

        # unpack the cache
        Z = self.cache["Z"]
        W = self.parameters["W"]
        X = self.cache["X"]
        X_padded = self.cache["X_p"]
        p = self.cache["p"]
        out_rows, out_cols = self.cache["out_shape"]
        kernel_height, kernel_width, in_channels, out_channels = W.shape

        # compute the gradients of the loss w.r.t. all parameters as well as the input of the layer
        dLdZ = self.activation.backward(Z, dLdY)
        dLdb = np.einsum('ijkl->il', dLdZ)

        dLdW = np.zeros(W.shape)
        # for k1 in range(kernel_height):
        #     for k2 in range(kernel_width):
        #         for m2 in range(out_channels):
        #             dldz = dLdZ[:, :, :, m2]
        #             x = X[:, k1:k1 + out_rows, k2:k2 + out_cols, :]
        #             temp1 = np.einsum('...jk, ...jkm', dldz, x)
        #             temp2 = np.einsum('iij', temp1)
        #             dLdW[k1, k2, :, m2] = temp2

        dLdX_padded = np.zeros(X_padded.shape)
        W_flip = np.flip(np.flip(W, axis=0), axis=1)
        dLdZpd = convolution.pad_dilate2d(dLdZ, kernel_height, kernel_width, self.stride)
        for h in range(X_padded.shape[1]):
            for w in range(X_padded.shape[2]):
                dldz = dLdZpd[:, h:h + kernel_height, w:w + kernel_width, :]
                for k in range(in_channels):
                    w_flip = W_flip[:, :, k, :]
                    dLdX_padded[:, h, w, k] = np.einsum('ijkl, jkl', dldz, w_flip)

        dLdX = dLdX_padded[:, p[0]:dLdX_padded.shape[1]-p[0], p[2]:dLdX_padded.shape[2]-p[2], :]

        # store the gradients in `self.gradients`
        # self.gradients["W"] = dLdW
        self.gradients["b"] = dLdb
        self.gradients["X"] = dLdX

        return dLdX


class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(self, kernel_shape: Tuple[int, int], mode: str = "max", stride: int = 1,
                 pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """

        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_height, kernel_width = self.kernel_shape

        X_pad, p = convolution.pad2d(X, self.pad, self.kernel_shape, stride=self.stride)
        out_rows = (in_rows - kernel_height + 2 * self.pad[0]) // self.stride + 1
        out_cols = (in_cols - kernel_width + 2 * self.pad[1]) // self.stride + 1
        Z = np.zeros((n_examples, out_rows, out_cols, in_channels))
        dLdX_pad = np.zeros(X_pad.shape)

        max_filter = np.zeros((n_examples, in_channels, out_rows, out_cols,  kernel_height, kernel_width))
        max_filter = max_filter.reshape((n_examples * in_channels, out_rows, out_cols, kernel_height * kernel_width))
        print(dLdX_pad.shape)
        print(max_filter.shape)
        for d1 in range(out_rows):
            d1_step = d1 * self.stride
            for d2 in range(out_cols):
                d2_step = d2 * self.stride
                x = X_pad[:, d1_step:d1_step + kernel_height, d2_step:d2_step + kernel_width, :]
                if self.mode == "max":
                    Z[:, d1, d2, :] = np.amax(x, axis=(1, 2))
                    x = x.reshape(X_pad.shape[0], kernel_height * kernel_width, X_pad.shape[3])
                    np.moveaxis(x, -1, 1)
                    x = x.reshape(X_pad.shape[0] * X_pad.shape[3], kernel_height * kernel_width)
                    b = np.argmax(x, axis=1)
                    # max_filter[:, d1, d2, b] = 1
                    # dLdX_pad[:, d1_step + a, d2_step + b, :] += 1

                else:  # average case
                    Z[:, d1, d2, :] = x.mean(axis=(1, 2))

        max_filter = max_filter.reshape((n_examples, in_channels, out_rows, out_cols, kernel_height, kernel_width))

        # cache any values required for backprop
        self.cache["Z"] = Z
        self.cache["X"] = X
        self.cache["X_pad"] = X_pad
        self.cache["out_rows"] = out_rows
        self.cache["out_cols"] = out_cols
        self.cache["p"] = p
        self.cache["max_filter"] = max_filter

        return Z

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """

        # unpack the cache
        Z = self.cache["Z"]
        X = self.cache["X"]
        X_pad = self.cache["X_pad"]
        p = self.cache["p"]
        max_filter = self.cache["max_filter"]
        out_rows, out_cols = self.cache["out_rows"], self.cache["out_cols"]
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_height, kernel_width = self.kernel_shape

        # perform a backward pass
        dLdZ = dLdY

        dLdX_pad = np.zeros(X_pad.shape)
        dLdZpd = convolution.pad_dilate2d(dLdZ, kernel_height, kernel_width, self.stride)
        avg_kernel = (1 / (kernel_height * kernel_width)) * np.ones((kernel_height, kernel_width))

        for h in range(X_pad.shape[1]):
            for w in range(X_pad.shape[2]):
                for k in range(X_pad.shape[3]):
                    dldz = dLdZpd[:, h:h + kernel_height, w:w + kernel_width, k]
                    if self.mode == "max":
                        filt = max_filter[:, k, h, w, :, :]
                        tot = np.einsum('ijk, ljk', dldz, filt)
                        diagg = np.einsum('ii', tot)
                        # print(diagg.shape)
                        dLdX_pad[:, h, w, k] = diagg
                    else:  # average case
                        dLdX_pad[:, h, w, k] = np.einsum('ijk, jk', dldz, avg_kernel)

        dLdX = dLdX_pad[:, p[0]:dLdX_pad.shape[1] - p[0], p[2]:dLdX_pad.shape[2] - p[2], :]

        return dLdX

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX
