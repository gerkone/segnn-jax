import warnings
from typing import Callable, Optional, Tuple, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax._src.tensor_products import naive_broadcast_decorator

from .config import config

InitFn = Callable[[str, Tuple[int, ...], float, jnp.dtype], jnp.ndarray]
TensorProductFn = Callable[[e3nn.IrrepsArray, e3nn.IrrepsArray], e3nn.IrrepsArray]


def uniform_init(
    name: str,
    path_shape: Tuple[int, ...],
    weight_std: float,
    dtype: jnp.dtype = config("default_dtype"),
) -> jnp.ndarray:
    return hk.get_parameter(
        name,
        shape=path_shape,
        dtype=dtype,
        init=hk.initializers.RandomUniform(minval=-weight_std, maxval=weight_std),
    )


class O3TensorProduct(hk.Module):
    """
    O(3) equivariant linear parametrized tensor product layer.

    Functionally the same as O3TensorProductLegacy, but around 5-10% faster.
    FullyConnectedTensorProduct seems faster than tensor_product + linear:
    https://github.com/e3nn/e3nn-jax/releases/tag/0.14.0
    """

    def __init__(
        self,
        output_irreps: e3nn.Irreps,
        *,
        biases: bool = True,
        name: Optional[str] = None,
        init_fn: Optional[InitFn] = None,
        gradient_normalization: Optional[Union[str, float]] = None,
        path_normalization: Optional[Union[str, float]] = None,
    ):
        """Initialize the tensor product.

        Args:
            output_irreps: Output representation
            biases: If set ot true will add biases
            name: Name of the linear layer params
            init_fn: Weight initialization function. Default is uniform.
            gradient_normalization: Gradient normalization method. Default is "path"
                NOTE: gradient_normalization="element" is the default in torch and haiku.
            path_normalization: Path normalization method. Default is "element"
        """
        super().__init__(name)

        if not isinstance(output_irreps, e3nn.Irreps):
            output_irreps = e3nn.Irreps(output_irreps)
        self.output_irreps = output_irreps

        # tp weight init
        if not init_fn:
            init_fn = uniform_init
        self.get_parameter = init_fn

        if not gradient_normalization:
            gradient_normalization = config("gradient_normalization")
        if not path_normalization:
            path_normalization = config("path_normalization")
        self._gradient_normalization = gradient_normalization
        self._path_normalization = path_normalization

        self.biases = biases and "0e" in self.output_irreps

    def _build_tensor_product(
        self, left_irreps: e3nn.Irreps, right_irreps: e3nn.Irreps
    ) -> Callable:
        """Build the tensor product function."""
        tp = e3nn.FunctionalFullyConnectedTensorProduct(
            left_irreps,
            right_irreps,
            self.output_irreps,
            gradient_normalization=self._gradient_normalization,
            path_normalization=self._path_normalization,
        )
        ws = [
            self.get_parameter(
                name=(
                    f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                    f"{tp.irreps_in1[ins.i_in1]},"
                    f"{tp.irreps_in2[ins.i_in2]},"
                    f"{tp.irreps_out[ins.i_out]}"
                ),
                path_shape=ins.path_shape,
                weight_std=ins.weight_std,
            )
            for ins in tp.instructions
        ]

        def tensor_product(x, y, **kwargs):
            return tp.left_right(ws, x, y, **kwargs)._convert(self.output_irreps)

        return naive_broadcast_decorator(tensor_product)

    def _build_biases(self) -> Callable:
        """Build the add bias function."""
        b = [
            self.get_parameter(
                f"b[{i_out}] {self.output_irreps}",
                path_shape=(mul_ir.dim,),
                weight_std=1 / jnp.sqrt(mul_ir.dim),
            )
            for i_out, mul_ir in enumerate(self.output_irreps)
            if mul_ir.ir.is_scalar()
        ]
        b = e3nn.IrrepsArray(f"{self.output_irreps.count('0e')}x0e", jnp.concatenate(b))

        # TODO: could be improved
        def _wrapper(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            scalars = x.filter("0e")
            other = x.filter(drop="0e")
            return e3nn.concatenate(
                [scalars + b.broadcast_to(scalars.shape), other], axis=1
            )

        return _wrapper

    def __call__(
        self, x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None, **kwargs
    ) -> e3nn.IrrepsArray:
        """Applies an O(3) equivariant linear parametrized tensor product layer.

        Args:
            x (IrrepsArray): Left tensor
            y (IrrepsArray): Right tensor. If None it defaults to np.ones.

        Returns:
            The output to the weighted tensor product (IrrepsArray).
        """

        if not y:
            y = e3nn.IrrepsArray("1x0e", jnp.ones((1, 1), dtype=x.dtype))

        if x.irreps.lmax == 0 and y.irreps.lmax == 0 and self.output_irreps.lmax > 0:
            warnings.warn(
                f"The specified output irreps ({self.output_irreps}) are not scalars "
                "but both operands are. This can have undesired behaviour (NaN). Try "
                "redistributing them into scalars or choose higher orders."
            )

        tp = self._build_tensor_product(x.irreps, y.irreps)
        output = tp(x, y, **kwargs)

        if self.biases:
            # add biases
            bias_fn = self._build_biases()
            return bias_fn(output)

        return output


def O3TensorProductLegacy(
    output_irreps: e3nn.Irreps,
    *,
    biases: bool = True,
    name: Optional[str] = None,
    init_fn: Optional[InitFn] = None,
    gradient_normalization: Optional[Union[str, float]] = "element",
    path_normalization: Optional[Union[str, float]] = None,
):
    """O(3) equivariant linear parametrized tensor product layer.
    Legacy version of O3TensorProduct that uses e3nn.haiku.Linear instead of
    e3nn.FunctionalFullyConnectedTensorProduct.

    Args:
        output_irreps: Output representation
        biases: If set ot true will add biases
        name: Name of the linear layer params
        init_fn: Weight initialization function. Default is uniform.
        gradient_normalization: Gradient normalization method. Default is "path"
            NOTE: gradient_normalization="element" is the default in torch and haiku.
        path_normalization: Path normalization method. Default is "element"

    Returns:
        A function that returns the output to the weighted tensor product.
    """

    if not isinstance(output_irreps, e3nn.Irreps):
        output_irreps = e3nn.Irreps(output_irreps)

    if not init_fn:
        init_fn = uniform_init

    linear = e3nn.haiku.Linear(
        output_irreps,
        get_parameter=init_fn,
        biases=biases,
        name=name,
        gradient_normalization=gradient_normalization,
        path_normalization=path_normalization,
    )

    def _tensor_product(
        x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None
    ) -> TensorProductFn:
        """Applies an O(3) equivariant linear parametrized tensor product layer.

        Args:
            x (IrrepsArray): Left tensor
            y (IrrepsArray): Right tensor. If None it defaults to np.ones.

        Returns:
            The output to the weighted tensor product (IrrepsArray).
        """

        if not y:
            y = e3nn.IrrepsArray("1x0e", jnp.ones((1, 1), dtype=x.dtype))

        if x.irreps.lmax == 0 and y.irreps.lmax == 0 and output_irreps.lmax > 0:
            warnings.warn(
                f"The specified output irreps ({output_irreps}) are not scalars "
                "but both operands are. This can have undesired behaviour (NaN). Try "
                "redistributing them into scalars or choose higher orders."
            )

        tp = e3nn.tensor_product(x, y)

        return linear(tp)

    return _tensor_product


O3Layer = O3TensorProduct if config("o3_layer") == "new" else O3TensorProductLegacy


def O3TensorProductGate(
    output_irreps: e3nn.Irreps,
    *,
    biases: bool = True,
    scalar_activation: Optional[Callable] = None,
    gate_activation: Optional[Callable] = None,
    name: Optional[str] = None,
    init_fn: Optional[InitFn] = None,
) -> TensorProductFn:
    """Non-linear (gated) O(3) equivariant linear tensor product layer.

    The tensor product lifts the input representation to have gating scalars.

    Args:
        output_irreps: Output representation
        biases: Add biases
        scalar_activation: Activation function for scalars
        gate_activation: Activation function for higher order
        name: Name of the linear layer params

    Returns:
        Function that applies the gated tensor product layer.
    """

    if not isinstance(output_irreps, e3nn.Irreps):
        output_irreps = e3nn.Irreps(output_irreps)

    # lift output with gating scalars
    gate_irreps = e3nn.Irreps(
        f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
    )
    tensor_product = O3Layer(
        (gate_irreps + output_irreps).regroup(),
        biases=biases,
        name=name,
        init_fn=init_fn,
    )
    if not scalar_activation:
        scalar_activation = jax.nn.silu
    if not gate_activation:
        gate_activation = jax.nn.sigmoid

    def _gated_tensor_product(
        x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None, **kwargs
    ) -> e3nn.IrrepsArray:
        tp = tensor_product(x, y, **kwargs)
        return e3nn.gate(tp, even_act=scalar_activation, odd_gate_act=gate_activation)

    return _gated_tensor_product
