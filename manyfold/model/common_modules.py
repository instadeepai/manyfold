# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of common Haiku modules for use in protein folding."""
import collections
import numbers
from typing import Optional, Sequence, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(0.87962566103423978, dtype=np.float32)


def get_initializer_scale(initializer_name, input_shape):
    """Get Initializer for weights and scale to multiply activations by."""

    if initializer_name == "zeros":
        w_init = hk.initializers.Constant(0.0)
    else:
        # fan-in scaling
        scale = 1.0
        for channel_dim in input_shape:
            scale /= channel_dim
        if initializer_name == "relu":
            scale *= 2

        noise_scale = scale

        stddev = np.sqrt(noise_scale)
        # Adjust stddev for truncation.
        stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
        w_init = hk.initializers.TruncatedNormal(mean=0.0, stddev=stddev)

    return w_init


class Linear(hk.Module):
    """Protein folding specific Linear module.

    This differs from the standard Haiku Linear in a few ways:
      * It supports inputs and outputs of arbitrary rank
      * Initializers are specified by strings
    """

    def __init__(
        self,
        num_output: Union[int, Sequence[int]],
        initializer: str = "linear",
        num_input_dims: int = 1,
        use_bias: bool = True,
        bias_init: float = 0.0,
        precision=None,
        name: str = "linear",
    ):
        """Constructs Linear Module.

        Args:
          num_output: Number of output channels. Can be tuple when outputting
              multiple dimensions.
          initializer: What initializer to use, should be one of {'linear', 'relu',
            'zeros'}
          num_input_dims: Number of dimensions from the end to project.
          use_bias: Whether to include trainable bias
          bias_init: Value used to initialize bias.
          precision: What precision to use for matrix multiplication, defaults
            to None.
          name: Name of module, used for name scopes.
        """
        super().__init__(name=name)
        if isinstance(num_output, numbers.Integral):
            self.output_shape = (num_output,)
        else:
            self.output_shape = tuple(num_output)
        self.initializer = initializer
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.num_input_dims = num_input_dims
        self.num_output_dims = len(self.output_shape)
        self.precision = precision

    def __call__(self, inputs):
        """Connects Module.

        Args:
          inputs: Tensor with at least num_input_dims dimensions.

        Returns:
          output of shape [...] + num_output.
        """

        if self.num_input_dims > 0:
            in_shape = inputs.shape[-self.num_input_dims :]
        else:
            in_shape = ()

        weight_init = get_initializer_scale(self.initializer, in_shape)

        in_letters = "abcde"[: self.num_input_dims]
        out_letters = "hijkl"[: self.num_output_dims]

        weight_shape = in_shape + self.output_shape
        weights = hk.get_parameter("weights", weight_shape, inputs.dtype, weight_init)

        equation = f"...{in_letters}, {in_letters}{out_letters}->...{out_letters}"

        output = jnp.einsum(equation, inputs, weights, precision=self.precision)

        if self.use_bias:
            bias = hk.get_parameter(
                "bias",
                self.output_shape,
                inputs.dtype,
                hk.initializers.Constant(self.bias_init),
            )
            output += bias

        return output


class LayerNorm(hk.Module):
    """LayerNorm module.
    See: https://arxiv.org/abs/1607.06450.
    """

    def __init__(
        self,
        axis: Union[int, Sequence[int], slice],
        create_scale: bool,
        create_offset: bool,
        eps: float = 1e-5,
        scale_init: Optional[hk.initializers.Initializer] = None,
        offset_init: Optional[hk.initializers.Initializer] = None,
        use_fast_variance: bool = False,
        name: Optional[str] = None,
    ):
        """Constructs a LayerNorm module.
        Args:
          axis: Integer, list of integers, or slice indicating which axes to
            normalize over.
          create_scale: Bool, defines whether to create a trainable scale
            per channel applied after the normalization.
          create_offset: Bool, defines whether to create a trainable offset
            per channel applied after normalization and scaling.
          eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
            as in the paper and Sonnet.
          scale_init: Optional initializer for gain (aka scale). By default, one.
          offset_init: Optional initializer for bias (aka offset). By default, zero.
          use_fast_variance: If true, use a faster but less numerically stable
            formulation for computing variance.
          name: The module name.
        """
        super().__init__(name=name)
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

        if isinstance(axis, slice):
            self.axis = axis
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, collections.abc.Iterable) and all(
            isinstance(ax, int) for ax in axis
        ):
            self.axis = tuple(axis)
        else:
            raise ValueError("`axis` should be an int, slice or iterable of ints.")

        self.eps = eps
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros
        self.use_fast_variance = use_fast_variance

    def __call__(
        self,
        inputs: jnp.ndarray,
        scale: Optional[jnp.ndarray] = None,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Connects the layer norm.
        Args:
          inputs: An array, where the data format is ``[N, ..., C]``.
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.
        Returns:
          The array, normalized.
        """
        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`."
            )

        axis = self.axis
        if isinstance(axis, slice):
            axis = tuple(range(inputs.ndim)[axis])

        mean = jnp.mean(inputs, axis=axis, keepdims=True)
        if self.use_fast_variance:
            mean_of_squares = jnp.mean(jnp.square(inputs), axis=axis, keepdims=True)
            variance = mean_of_squares - jnp.square(mean)
        else:
            variance = jnp.var(inputs, axis=axis, keepdims=True)

        param_shape = inputs.shape[-1:]
        if self.create_scale:
            scale = hk.get_parameter(
                "scale", param_shape, inputs.dtype, init=self.scale_init
            )
        elif scale is None:
            scale = np.array(1.0, dtype=inputs.dtype)

        if self.create_offset:
            offset = hk.get_parameter(
                "offset", param_shape, inputs.dtype, init=self.offset_init
            )
        elif offset is None:
            offset = np.array(0.0, dtype=inputs.dtype)

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        eps = jax.lax.convert_element_type(self.eps, variance.dtype)
        inv = scale * jax.lax.rsqrt(variance + eps)
        return inv * (inputs - mean) + offset
