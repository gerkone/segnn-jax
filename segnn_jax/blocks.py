import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax.experimental import linear_shtp as escn
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct

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


class TensorProduct(hk.Module, ABC):
    """O(3) equivariant linear parametrized tensor product layer."""

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
        super().__init__(name=name)

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

    def _check_input(
        self, x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        if not y:
            y = e3nn.IrrepsArray("1x0e", jnp.ones((1, 1), dtype=x.dtype))

        if x.irreps.lmax == 0 and y.irreps.lmax == 0 and self.output_irreps.lmax > 0:
            warnings.warn(
                f"The specified output irreps ({self.output_irreps}) are not scalars "
                "but both operands are. This can have undesired behaviour (NaN). Try "
                "redistributing them into scalars or choose higher orders."
            )

        return x, y

    @abstractmethod
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
        raise NotImplementedError


class O3TensorProduct(TensorProduct):
    """O(3) equivariant linear parametrized tensor product layer.

    Original O3TensorProduct version that uses tensor_product + Linear instead of
    FullyConnectedTensorProduct.
    From e3nn 0.19.2 (https://github.com/e3nn/e3nn-jax/releases/tag/0.19.2), this is
    as fast as FullyConnectedTensorProduct.
    """

    def __init__(
        self,
        output_irreps: e3nn.Irreps,
        *,
        biases: bool = True,
        name: Optional[str] = None,
        init_fn: Optional[InitFn] = None,
        gradient_normalization: Optional[Union[str, float]] = "element",
        path_normalization: Optional[Union[str, float]] = None,
    ):
        super().__init__(
            output_irreps,
            biases=biases,
            name=name,
            init_fn=init_fn,
            gradient_normalization=gradient_normalization,
            path_normalization=path_normalization,
        )

        self._linear = e3nn.haiku.Linear(
            self.output_irreps,
            get_parameter=self.get_parameter,
            biases=self.biases,
            name=f"{self.name}_linear",
            gradient_normalization=self._gradient_normalization,
            path_normalization=self._path_normalization,
        )

    def __call__(
        self, x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None
    ) -> TensorProductFn:
        x, y = self._check_input(x, y)
        # tensor product + linear
        tp = self._linear(e3nn.tensor_product(x, y))
        return tp


class O3TensorProductFC(TensorProduct):
    """
    O(3) equivariant linear parametrized tensor product layer.

    Functionally the same as O3TensorProduct, but uses FullyConnectedTensorProduct and
    is slightly slower (~5-10%) than tensor_prodict + Linear.
    """

    def _build_tensor_product(
        self, left_irreps: e3nn.Irreps, right_irreps: e3nn.Irreps
    ) -> Callable:
        """Build the tensor product function."""
        tp = FunctionalFullyConnectedTensorProduct(
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

        def _tensor_product(x, y, **kwargs):
            return tp.left_right(ws, x, y, **kwargs).rechunk(self.output_irreps)

        # naive broadcasting wrapper
        # TODO: not the best
        def _tp_wrapper(*args):
            leading_shape = jnp.broadcast_shapes(*(arg.shape[:-1] for arg in args))
            args = [arg.broadcast_to(leading_shape + (-1,)) for arg in args]
            for _ in range(len(leading_shape)):
                f = jax.vmap(_tensor_product)
            return f(*args)

        return _tp_wrapper

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
        def _bias_wrapper(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            scalars = x.filter("0e")
            other = x.filter(drop="0e")
            return e3nn.concatenate(
                [scalars + b.broadcast_to(scalars.shape), other], axis=1
            )

        return _bias_wrapper

    def __call__(
        self, x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None, **kwargs
    ) -> e3nn.IrrepsArray:
        x, y = self._check_input(x, y)

        tp = self._build_tensor_product(x.irreps, y.irreps)
        output = tp(x, y, **kwargs)

        if self.biases:
            # add biases
            bias_fn = self._build_biases()
            return bias_fn(output)

        return output


class O3TensorProductSCN(TensorProduct):
    """
    O(3) equivariant linear parametrized tensor product layer.

    O3TensorProduct with eSCN optimization for larger spherical harmonic orders. Should
    be used without spherical harmonics on the inputs.
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
        super().__init__(
            output_irreps,
            biases=biases,
            name=name,
            init_fn=init_fn,
            gradient_normalization=gradient_normalization,
            path_normalization=path_normalization,
        )

        self._linear = e3nn.haiku.Linear(
            self.output_irreps,
            get_parameter=self.get_parameter,
            biases=self.biases,
            name=f"{self.name}_linear",
            gradient_normalization=self._gradient_normalization,
            path_normalization=self._path_normalization,
        )

    def __call__(
        self, x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None, **kwargs
    ) -> e3nn.IrrepsArray:
        """Apply the layer. y must not be into spherical harmonics."""
        x, y = self._check_input(x, y)
        shtp = e3nn.utils.vmap(escn.shtp, in_axes=(0, 0, None))
        tp = shtp(x, y, self.output_irreps)
        return self._linear(tp)


O3_LAYERS = {
    "tpl": O3TensorProduct,
    "fctp": O3TensorProductFC,
    "scn": O3TensorProductSCN,
}


def O3TensorProductGate(
    output_irreps: e3nn.Irreps,
    *,
    biases: bool = True,
    scalar_activation: Optional[Callable] = None,
    gate_activation: Optional[Callable] = None,
    name: Optional[str] = None,
    init_fn: Optional[InitFn] = None,
    o3_layer: Optional[Union[str, TensorProduct]] = None,
) -> TensorProductFn:
    """Non-linear (gated) O(3) equivariant linear tensor product layer.

    The tensor product lifts the input representation to have gating scalars.

    Args:
        output_irreps: Output representation
        biases: Add biases
        scalar_activation: Activation function for scalars
        gate_activation: Activation function for higher order
        name: Name of the linear layer params
        o3_layer: Tensor product layer type. "tpl", "fctp", "scn" or a custom layer

    Returns:
        Function that applies the gated tensor product layer
    """

    if not isinstance(output_irreps, e3nn.Irreps):
        output_irreps = e3nn.Irreps(output_irreps)

    # lift output with gating scalars
    gate_irreps = e3nn.Irreps(
        f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
    )

    if o3_layer is None:
        o3_layer = config("o3_layer")

    if isinstance(o3_layer, str):
        assert o3_layer in O3_LAYERS, f"Unknown O3 layer {o3_layer}."
        O3Layer = O3_LAYERS[o3_layer]
    else:
        O3Layer = o3_layer

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
