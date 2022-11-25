import warnings
from typing import Callable, Optional

import e3nn_jax as e3nn
import jax

from .graphs import SteerableGraphsTuple


def O3TensorProduct(
    x: e3nn.IrrepsArray,
    y: e3nn.IrrepsArray,
    output_irreps: e3nn.Irreps,
    biases: bool = True,
    name: Optional[str] = None,
) -> e3nn.IrrepsArray:
    """Applies a linear parametrized tensor product layer.

    Args:
        x (IrrepsArray): Left tensor
        y (IrrepsArray): Right tensor
        output_irreps: Output representation
        biases: Specifies tu use biases
        name: Name of the linear layer params

    Returns:
        The output to the weighted tensor product (IrrepsArray).
    """
    if x.irreps.lmax == 0 and y.irreps.lmax == 0 and output_irreps.lmax > 0:
        warnings.warn(
            f"The specified output irreps ({output_irreps}) are not scalars but "
            "both operands are. This can have undesired behaviour such as null "
            "output Try redistributing them into scalars or choosing higher "
            "order for the operands."
        )

    tp = e3nn.tensor_product(x, y, irrep_normalization="component")
    # NOTE gradient_normalization="element" for 1/sqrt(fanin) initialization
    #  this is the default in torch and is similar to what is used in the
    #  original code (tp_rescale)
    return e3nn.Linear(
        output_irreps, biases=biases, gradient_normalization="element", name=name
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
        biases: Specifies tu use biases
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
    graph: SteerableGraphsTuple,
    node_attributes: e3nn.IrrepsArray,
    embed_irreps: e3nn.Irreps,
    edge_attributes: Optional[e3nn.IrrepsArray] = None,
) -> SteerableGraphsTuple:
    """Linear steerable embedding for the graph nodes.

    Embeds the graph nodes in the representation space :param embed_irreps:.

    Args:
        graph (SteerableGraphsTuple): Input graph
        node_attributes (IrrepsArray): Steerable node attributes
        embed_irreps: Output representation
        edge_attributes (IrrepsArray): Steerable edge attributes

    Returns:
        The graph with replaced node embedding.
    """
    nodes = O3TensorProduct(
        graph.nodes, node_attributes, embed_irreps, name="o3_mbedding"
    )
    graph = graph._replace(nodes=nodes)

    if edge_attributes:
        edges = O3TensorProduct(
            graph.edges, edge_attributes, embed_irreps, name="o3_mbedding_edges"
        )
        graph = graph._replace(edges=edges)

    return graph
