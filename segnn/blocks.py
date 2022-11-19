from typing import Callable, Optional

import e3nn_jax as e3nn
import jax
import jraph


def O3TensorProduct(
    x: e3nn.IrrepsArray,
    y: e3nn.IrrepsArray,
    output_irreps: e3nn.Irreps,
    biases: bool = True,
) -> e3nn.IrrepsArray:
    tp = e3nn.tensor_product(x, y, irrep_normalization="component")
    return e3nn.Linear(output_irreps, biases=biases)(tp)


def O3TensorProductGate(
    x: e3nn.IrrepsArray,
    y: e3nn.IrrepsArray,
    output_irreps: e3nn.Irreps,
    biases: bool = True,
    scalar_activation: Optional[Callable] = None,
    gate_activation: Optional[Callable] = None,
) -> e3nn.IrrepsArray:

    if scalar_activation is None:
        scalar_activation = jax.nn.silu
    if gate_activation is None:
        gate_activation = jax.nn.sigmoid

    tp = O3TensorProduct(x, y, output_irreps, biases)
    return e3nn.gate(tp, even_act=scalar_activation, odd_gate_act=gate_activation)


def O3Embedding(
    graph: jraph.GraphsTuple,
    node_attributes: e3nn.IrrepsArray,
    output_irreps: e3nn.Irreps,
    biases: bool = True,
) -> jraph.GraphsTuple:
    nodes = O3TensorProduct(
        graph.nodes,
        node_attributes,
        output_irreps,
        biases,
    )
    return graph._replace(nodes=nodes)
