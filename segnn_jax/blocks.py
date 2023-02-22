import warnings
from typing import Any, Callable, Optional, Tuple, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax._src.tensor_products import naive_broadcast_decorator

from .graph_utils import SteerableGraphsTuple

InitFn = Callable[[str, Tuple[int, ...], float, Any], jnp.ndarray]


class O3TensorProduct(hk.Module):
    """O(3) equivariant linear parametrized tensor product layer.

    Attributes:
        output_irreps: Output representation
        biases: If set ot true will add biases
        name: Name of the linear layer params
        init_fn: Weight initialization function
        gradient_normalization: Gradient normalization method. Default is "path"
            NOTE: gradient_normalization="element" for 1/sqrt(fanin) initialization.
            This is the default in torch and is similar to what is used in the original
            code (tp_rescale).
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
        gradient_normalization: Optional[Union[str, float]] = "element",
        path_normalization: Optional[Union[str, float]] = None,
    ):
        super().__init__(name)

        self.output_irreps = output_irreps.regroup()
        if not right_irreps:
            right_irreps = e3nn.Irreps("1x0e")
        self.right_irreps = right_irreps.regroup()
        self.left_irreps = left_irreps.regroup()

        self._biases = biases

        if not init_fn:

            def init_fn(
                name: str,
                path_shape: Tuple[int, ...],
                weight_std: float,
                dtype: jnp.dtype = jnp.float32,
            ):
                return hk.get_parameter(
                    name,
                    shape=path_shape,
                    dtype=dtype,
                    init=hk.initializers.RandomNormal(stddev=weight_std),
                )

        self._init_fn = init_fn

        # NOTE gradient_normalization="element" for 1/sqrt(fanin) initialization
        #  this is the default in torch and is similar to what is used in the
        # original code (tp_rescale). On deeper networks could also init with uniform.
        self._gradient_normalization = gradient_normalization
        self._path_normalization = path_normalization

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
            y = e3nn.IrrepsArray("1x0e", jnp.ones((x.shape[0], 1)))

        if x.irreps.lmax == 0 and y.irreps.lmax == 0 and self._output_irreps.lmax > 0:
            warnings.warn(
                f"The specified output irreps ({self.output_irreps}) are not scalars but both "
                "operands are. This can have undesired behaviour such as null output Try "
                "redistributing them into scalars or chose higher orders for the operands."
            )

        x = x.remove_nones().regroup()
        y = y.remove_nones().regroup()

        assert (
            x.irreps == self.left_irreps
        ), f"Left irreps do not match. Got {x.irreps}, expected {self.left_irreps}"
        assert (
            y.irreps == self.right_irreps
        ), f"Right irreps do not match. Got {y.irreps}, expected {self.right_irreps}"

        # NOTE FunctionalFullyConnectedTensorProduct appears to be faster than combining
        #  tensor_product+linear: https://github.com/e3nn/e3nn-jax/releases/tag/0.14.0
        #  Implementation adapted from e3nn.haiku.FullyConnectedTensorProduct

        tp = e3nn.FunctionalFullyConnectedTensorProduct(
            x.irreps,
            y.irreps,
            self.output_irreps.simplify(),
            irrep_normalization="component",
            gradient_normalization=self._gradient_normalization,
            path_normalization=self._path_normalization,
        )

        ws = [
            self._init_fn(
                name=(
                    f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                    f"{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}"
                ),
                path_shape=ins.path_shape,
                weight_std=ins.weight_std,
                dtype=x.dtype,
            )
            for ins in tp.instructions
        ]

        f = naive_broadcast_decorator(
            lambda x1, y1: tp.left_right(ws, x1, y1, **kwargs)
        )
        output = f(x, y)._convert(self.output_irreps)

        if self._biases and "0e" in self.output_irreps:
            # add biases
            b = [
                self._init_fn(
                    f"b[{i_out}] {tp.irreps_out[i_out]}",
                    (mul_ir.dim,),
                    0.0,
                    x.dtype,
                )
                for i_out, mul_ir in enumerate(self.output_irreps)
                if mul_ir.ir.is_scalar()
            ]
            b = e3nn.IrrepsArray(
                f"{self.output_irreps.count('0e')}x0e", jnp.concatenate(b)
            )
            output = e3nn.concatenate(
                [output.filter("0e") + b, output.filter(drop="0e")], axis=1
            )

        return output


class O3TensorProductGate(O3TensorProduct):
    """Non-linear (gated) O(3) equivariant linear tensor product layer.

    The tensor product lifts the input representation to have gating scalars.

    Attributes:
        output_irreps: Output representation
        biases: Add biases
        scalar_activation: Activation function for scalars
        gate_activation: Activation function for higher order
        name: Name of the linear layer params
    """

    def __init__(
        self,
        output_irreps: e3nn.Irreps,
        *,
        left_irreps: e3nn.Irreps,
        right_irreps: Optional[e3nn.Irreps] = None,
        biases: bool = True,
        scalar_activation: Optional[Callable] = None,
        gate_activation: Optional[Callable] = None,
        name: Optional[str] = None,
        init_fn: Optional[InitFn] = None,
    ):
        super().__init__(
            output_irreps,
            left_irreps=left_irreps,
            right_irreps=right_irreps,
            biases=biases,
            name=name,
            init_fn=init_fn,
        )
        if scalar_activation is None:
            self.scalar_activation = jax.nn.silu
        if gate_activation is None:
            self.gate_activation = jax.nn.sigmoid

        # lift input with gating scalars
        self.gate_irreps = e3nn.Irreps(
            f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
        )

    def __call__(
        self, x: e3nn.IrrepsArray, y: Optional[e3nn.IrrepsArray] = None, **kwargs
    ) -> e3nn.IrrepsArray:
        tp = super().__call__(x, y, **kwargs)
        return e3nn.gate(
            tp, even_act=self.scalar_activation, odd_gate_act=self.gate_activation
        )


class O3Embedding(hk.Module):
    """Linear steerable embedding.

    Embeds the graph nodes in the representation space :param embed_irreps:.

    Attributes:
        embed_irreps: Output representation
        embed_msg_features: If true also embed edges/message passing features
    """

    def __init__(self, embed_irreps: e3nn.Irreps, embed_msg_features: bool = True):
        super().__init__()
        self.embed_irreps = embed_irreps
        self.embed_msg_features = embed_msg_features

    def __call__(
        self,
        st_graph: SteerableGraphsTuple,
    ) -> SteerableGraphsTuple:
        """Apply linear steerable embedding.

        Embeds the graph nodes in the representation space :param embed_irreps:.

        Args:
            st_graph (SteerableGraphsTuple): Input graph

        Returns:
            The graph with replaced node embedding.
        """
        # TODO update
        graph = st_graph.graph
        nodes = O3TensorProduct(
            self.embed_irreps,
            left_irreps=graph.nodes.irreps,
            right_irreps=st_graph.node_attributes.irreps,
            name="o3_embedding_nodes",
        )(graph.nodes, st_graph.node_attributes)
        st_graph = st_graph._replace(graph=graph._replace(nodes=nodes))

        if self.embed_msg_features:
            additional_message_features = O3TensorProduct(
                self.embed_irreps,
                left_irreps=graph.nodes.irreps,
                right_irreps=st_graph.node_attributes.irreps,
                name="o3_embedding_msg_features",
            )(
                st_graph.additional_message_features,
                st_graph.edge_attributes,
            )
            st_graph = st_graph._replace(
                additional_message_features=additional_message_features
            )

        return st_graph
