import warnings
from typing import Any, Callable, Optional, Tuple, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax._src.tensor_products import naive_broadcast_decorator

from .config import config
from .graph_utils import SteerableGraphsTuple

InitFn = Callable[[str, Tuple[int, ...], float, Any], jnp.ndarray]
TensorProductFn = Callable[
    [e3nn.IrrepsArray, Optional[e3nn.IrrepsArray]], e3nn.IrrepsArray
]


def uniform_init(
    name: str,
    path_shape: Tuple[int, ...],
    weight_std: float,
    dtype: jnp.dtype = config("default_dtype"),
):
    return hk.get_parameter(
        name,
        shape=path_shape,
        dtype=dtype,
        init=hk.initializers.RandomUniform(minval=-weight_std, maxval=weight_std),
    )


class O3TensorProduct(hk.Module):
    """O(3) equivariant linear parametrized tensor product layer.

    Attributes:
        output_irreps: Output representation
        biases: If set ot true will add biases
        name: Name of the linear layer params
        init_fn: Weight initialization function. Default is uniform.
        gradient_normalization: Gradient normalization method. Default is "path"
            NOTE: gradient_normalization="element" is the default in torch and haiku.
        path_normalization: Path normalization method. Default is "element"
    """

    def __init__(
        self,
        output_irreps: e3nn.Irreps,
        *,
        left_irreps: e3nn.Irreps,
        right_irreps: Optional[e3nn.Irreps] = None,
        biases: bool = True,
        name: Optional[str] = None,
        init_fn: Optional[InitFn] = None,
        gradient_normalization: Optional[Union[str, float]] = None,
        path_normalization: Optional[Union[str, float]] = None,
    ):
        super().__init__(name)

        self.output_irreps = output_irreps
        if not right_irreps:
            right_irreps = e3nn.Irreps("1x0e")
        self.right_irreps = right_irreps
        self.left_irreps = left_irreps

        if not init_fn:
            init_fn = uniform_init

        self._get_parameter = init_fn

        if not gradient_normalization:
            gradient_normalization = config("gradient_normalization")
        if not path_normalization:
            path_normalization = config("path_normalization")

        # NOTE FunctionalFullyConnectedTensorProduct appears to be faster than combining
        #  tensor_product+linear: https://github.com/e3nn/e3nn-jax/releases/tag/0.14.0
        #  Implementation adapted from e3nn.haiku.FullyConnectedTensorProduct
        tp = e3nn.FunctionalFullyConnectedTensorProduct(
            left_irreps,
            right_irreps,
            output_irreps,
            gradient_normalization=gradient_normalization,
            path_normalization=path_normalization,
        )
        ws = [
            self._get_parameter(
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
            return tp.left_right(ws, x, y, **kwargs)._convert(output_irreps)

        self.tensor_product = naive_broadcast_decorator(tensor_product)
        self.biases = None

        if biases and "0e" in self.output_irreps:
            # add biases
            b = [
                self._get_parameter(
                    f"b[{i_out}] {tp.irreps_out[i_out]}",
                    path_shape=(mul_ir.dim,),
                    weight_std=1 / jnp.sqrt(mul_ir.dim),
                )
                for i_out, mul_ir in enumerate(output_irreps)
                if mul_ir.ir.is_scalar()
            ]
            b = e3nn.IrrepsArray(
                f"{self.output_irreps.count('0e')}x0e", jnp.concatenate(b)
            )

            self.biases = lambda x: e3nn.concatenate(
                [x.filter("0e") + b, x.filter(drop="0e")], axis=1
            )

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
            y = e3nn.IrrepsArray("1x0e", jnp.ones((1, 1)))

        if x.irreps.lmax == 0 and y.irreps.lmax == 0 and self.output_irreps.lmax > 0:
            warnings.warn(
                f"The specified output irreps ({self.output_irreps}) are not scalars but both "
                "operands are. This can have undesired behaviour such as null output Try "
                "redistributing them into scalars or chose higher orders for the operands."
            )

        assert (
            x.irreps == self.left_irreps
        ), f"Left irreps do not match. Got {x.irreps}, expected {self.left_irreps}"
        assert (
            y.irreps == self.right_irreps
        ), f"Right irreps do not match. Got {y.irreps}, expected {self.right_irreps}"

        output = self.tensor_product(x, y, **kwargs)

        if self.biases:
            # add biases
            return self.biases(output)

        return output


def O3TensorProductLegacy(
    output_irreps: e3nn.Irreps,
    *,
    left_irreps: e3nn.Irreps,
    right_irreps: Optional[e3nn.Irreps] = None,
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


    if not right_irreps:
        right_irreps = e3nn.Irreps("1x0e")

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
            y = e3nn.IrrepsArray("1x0e", jnp.ones((1, 1)))

        if x.irreps.lmax == 0 and y.irreps.lmax == 0 and output_irreps.lmax > 0:
            warnings.warn(
                f"The specified output irreps ({output_irreps}) are not scalars but both "
                "operands are. This can have undesired behaviour such as null output Try "
                "redistributing them into scalars or chose higher orders for the operands."
            )

        assert (
            x.irreps == left_irreps
        ), f"Left irreps do not match. Got {x.irreps}, expected {left_irreps}"
        assert (
            y.irreps == right_irreps
        ), f"Right irreps do not match. Got {y.irreps}, expected {right_irreps}"

        tp = e3nn.tensor_product(x, y)

        return linear(tp)

    return _tensor_product


O3Layer = O3TensorProduct if config("o3_layer") == "new" else O3TensorProductLegacy


def O3TensorProductGate(
    output_irreps: e3nn.Irreps,
    *,
    left_irreps: e3nn.Irreps,
    right_irreps: Optional[e3nn.Irreps] = None,
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
    # lift output with gating scalars
    gate_irreps = e3nn.Irreps(
        f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
    )
    tensor_product = O3Layer(
        (gate_irreps + output_irreps).regroup(),
        left_irreps=left_irreps,
        right_irreps=right_irreps,
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


def O3Embedding(embed_irreps: e3nn.Irreps, embed_edges: bool = True) -> Callable:
    """Linear steerable embedding.

    Embeds the graph nodes in the representation space :param embed_irreps:.

    Args:
        embed_irreps: Output representation
        embed_edges: If true also embed edges/message passing features

    Returns:
        Function to embed graph nodes (and optionally edges)
    """

    def _embedding(
        st_graph: SteerableGraphsTuple,
    ) -> SteerableGraphsTuple:
        # TODO update
        graph = st_graph.graph
        nodes = O3Layer(
            embed_irreps,
            left_irreps=graph.nodes.irreps,
            right_irreps=st_graph.node_attributes.irreps,
            name="embedding_nodes",
        )(graph.nodes, st_graph.node_attributes)
        st_graph = st_graph._replace(graph=graph._replace(nodes=nodes))

        if embed_edges:
            additional_message_features = O3Layer(
                embed_irreps,
                left_irreps=graph.nodes.irreps,
                right_irreps=st_graph.node_attributes.irreps,
                name="embedding_msg_features",
            )(
                st_graph.additional_message_features,
                st_graph.edge_attributes,
            )
            st_graph = st_graph._replace(
                additional_message_features=additional_message_features
            )

        return st_graph

    return _embedding
