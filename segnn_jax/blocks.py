import warnings
from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from .graph_utils import SteerableGraphsTuple


def _uniform_initialization(name: str, path_shape: Tuple[int, ...], weight_std: float):
    # initialize all params with uniform (also biases)
    return hk.get_parameter(
        name,
        shape=path_shape,
        init=hk.initializers.RandomUniform(-weight_std, weight_std),
    )


def O3TensorProduct(
    x: e3nn.IrrepsArray,
    y: Optional[e3nn.IrrepsArray],
    output_irreps: e3nn.Irreps,
    biases: bool = True,
    name: Optional[str] = None,
    uniform_initialization: bool = False,
) -> e3nn.IrrepsArray:
    """Applies a linear parametrized tensor product layer.

    Args:
        x (IrrepsArray): Left tensor
        y (IrrepsArray): Right tensor. If None it defaults to np.ones.
        output_irreps: Output representation
        biases: If set ot true will add biases
        name: Name of the linear layer params
        uniform_initialization: Initialize weights from a uniform distribution

    Returns:
        The output to the weighted tensor product (IrrepsArray).
    """

    if not y:
        y = e3nn.IrrepsArray("1x0e", jnp.ones((x.shape[0], 1)))

    if x.irreps.lmax == 0 and y.irreps.lmax == 0 and output_irreps.lmax > 0:
        warnings.warn(
            f"The specified output irreps ({output_irreps}) are not scalars but both "
            "operands are. This can have undesired behaviour such as null output Try "
            "redistributing them into scalars or chose higher orders for the operands."
        )

    tp = e3nn.tensor_product(x, y, irrep_normalization="component")
    # NOTE gradient_normalization="element" for 1/sqrt(fanin) initialization
    #  this is the default in torch and is similar to what is used in the
    #  original code (tp_rescale). On deeper networks could also init with uniform.
    return e3nn.Linear(
        output_irreps,
        biases=biases,
        gradient_normalization="element",
        name=name,
        get_parameter=(_uniform_initialization if uniform_initialization else None),
    )(tp)


def O3TensorProductGate(
    x: e3nn.IrrepsArray,
    y: e3nn.IrrepsArray,
    output_irreps: e3nn.Irreps,
    biases: bool = True,
    scalar_activation: Optional[Callable] = None,
    gate_activation: Optional[Callable] = None,
    name: Optional[str] = None,
) -> e3nn.IrrepsArray:
    """Applies a non-linear (gate) parametrized tensor product layer.

    The tensor product lifts the input representation to have gating scalars.

    Args:
        x (IrrepsArray): Left tensor
        y (IrrepsArray): Right tensor
        output_irreps: Output representation
        biases: Add biases
        scalar_activation: Activation function for scalars
        gate_activation: Activation function for higher order
        name: Name of the linear layer params

    Returns:
        The output to the weighted tensor product (IrrepsArray).
    """

    if scalar_activation is None:
        scalar_activation = jax.nn.silu
    if gate_activation is None:
        gate_activation = jax.nn.sigmoid

    # lift input with gating scalars
    gate_irreps = e3nn.Irreps(
        f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
    )

    tp = O3TensorProduct(
        x, y, (gate_irreps + output_irreps).regroup(), biases, name=name
    )
    return e3nn.gate(tp, even_act=scalar_activation, odd_gate_act=gate_activation)


def O3Embedding(
    st_graph: SteerableGraphsTuple,
    embed_irreps: e3nn.Irreps,
    embed_msg_features: bool = True,
) -> SteerableGraphsTuple:
    """Linear steerable embedding for the graph nodes.

    Embeds the graph nodes in the representation space :param embed_irreps:.

    Args:
        st_graph (SteerableGraphsTuple): Input graph
        embed_irreps: Output representation
        embed_msg_features: If true also embed edges/message passing features

    Returns:
        The graph with replaced node embedding.
    """
    graph = st_graph.graph
    nodes = O3TensorProduct(
        graph.nodes, st_graph.node_attributes, embed_irreps, name="o3_embedding_nodes"
    )
    st_graph = st_graph._replace(graph=graph._replace(nodes=nodes))

    if embed_msg_features:
        additional_message_features = O3TensorProduct(
            st_graph.additional_message_features,
            st_graph.edge_attributes,
            embed_irreps,
            name="o3_embedding_msg_features",
        )
        st_graph = st_graph._replace(
            additional_message_features=additional_message_features
        )

    return st_graph
