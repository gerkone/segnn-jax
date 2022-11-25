from typing import Callable, List, Optional, Tuple, Union

import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph

from .blocks import O3Embedding, O3TensorProduct, O3TensorProductGate
from .graphs import (
    SteerableGraphsTuple,
    SteerableMessagePassing,
    batched_graph_nodes,
    pooling,
)


def SEDecoder(
    latent_irreps: e3nn.Irreps,
    output_irreps: e3nn.Irreps,
    blocks: int = 1,
    task: str = "graph",
    pool: Optional[str] = "avg",
):
    r"""Steerable E(3) pooler and decoder.

    Args:
        latent_irreps: Representation from the previous block
        output_irreps: Output representation
        blocks: Number of tensor product blocks in the decoder
        task: Specifies where the output is located. Either 'graph' or 'node'

    Returns:
        Decoded latent feature space to output space.
    """

    assert task in ["node", "graph"]
    assert pool in ["avg", "sum", "none", None]

    def _ApplySEDecoder(graph: SteerableGraphsTuple):
        nodes = graph.nodes
        # pre pool block
        for i in range(blocks):
            nodes = O3TensorProductGate(
                nodes, graph.node_attributes, latent_irreps, name=f"prepool_{i}"
            )

        if task == "graph":
            # pool over graph
            pooled_irreps = (latent_irreps.num_irreps * output_irreps).regroup().irreps
            nodes = O3TensorProduct(
                nodes, graph.node_attributes, pooled_irreps, name=f"prepool_{blocks}"
            )

            # pooling layer
            if pool == "avg":
                pool_fn = jraph.segment_mean
            if pool == "sum":
                pool_fn = jraph.segment_sum
            nodes_to_graph, n_graphs = batched_graph_nodes(graph)
            nodes = pooling(graph, nodes_to_graph, n_graphs, aggregate_fn=pool_fn)

            # post pool
            for i in range(blocks):
                nodes = O3TensorProductGate(
                    nodes, graph.node_attributes, latent_irreps, name=f"postpool_{i}"
                )

        nodes = O3TensorProduct(
            nodes, graph.node_attributes, output_irreps, name="output"
        )

        return nodes

    return _ApplySEDecoder


def SEGNNLayer(
    output_irreps: e3nn.Irreps,
    layer_num: int,
    blocks: int = 2,
    norm: Optional[str] = None,
) -> Tuple[Callable, Callable]:
    r"""Steerable E(3) equivariant layer
    Args:
        output_irreps: Layer output representation
        layer_num: Numbering of the layer
        blocks: Number of tensor product blocks in the layer
        norm: Normalization type. Either be None, 'instance' or 'batch'

    Returns:
        Two function, message and node update mlps respectively.
    """

    assert norm in ["batch", "instance", "none", None]

    def _message(
        edge_attribute: e3nn.IrrepsArray,
        sender_nodes: e3nn.IrrepsArray,
        receiver_nodes: e3nn.IrrepsArray,
        edge_features: Optional[e3nn.IrrepsArray] = None,
    ) -> e3nn.IrrepsArray:
        # create messages
        msg = e3nn.concatenate([sender_nodes, receiver_nodes], axis=-1)
        if edge_features is not None:
            msg = e3nn.concatenate([msg, edge_features], axis=-1)
        # message mlp (phi_m in the paper) steered by edge attributeibutes
        for i in range(blocks):
            msg = O3TensorProductGate(
                msg, edge_attribute, output_irreps, name=f"message_{i}_{layer_num}"
            )
        if norm == "batch":
            msg = e3nn.BatchNorm(irreps=output_irreps)(msg)
        return msg

    def _update(
        node_attribute: e3nn.IrrepsArray,
        nodes: e3nn.IrrepsArray,
        msg: e3nn.IrrepsArray,
    ) -> e3nn.IrrepsArray:
        x = e3nn.concatenate((nodes, msg), axis=-1)
        # update mlp (phi_f in the paper) steered by node attributeibutes
        for i in range(blocks - 1):
            x = O3TensorProductGate(
                x, node_attribute, output_irreps, name=f"update_{i}_{layer_num}"
            )
        # last update layer without activation
        update = O3TensorProduct(
            x, node_attribute, output_irreps, name=f"update_{blocks}_{layer_num}"
        )
        # residual connection
        nodes += update
        if norm == "batch":
            nodes = e3nn.BatchNorm(irreps=output_irreps)(nodes)
        if norm == "instance":
            raise NotImplementedError("Instance norm not yet implemented")
        return nodes

    return _message, _update


def SEGNN(
    hidden_irreps: Union[List[e3nn.Irreps], e3nn.Irreps],
    output_irreps: e3nn.Irreps,
    num_layers: int,
    norm: Optional[str] = None,
    pool: Optional[str] = "avg",
    task: Optional[str] = "graph",
    blocks_per_layer: int = 2,
):
    r"""
    Steerable E(3) equivariant network.

    Original paper https://arxiv.org/abs/2110.02905.

    Args:
        hidden_irreps: Feature representation in the hidden layers
        output_irreps: Output representation.
        num_layers: Number of message passing layers
        norm: Normalization type. Either be None, 'instance' or 'batch'
        pool: Pooling mode (only for graph-wise tasks)
        task: Specifies where the output is located. Either 'graph' or 'node'
        blocks_per_layer: Number of tensor product blocks in each message passing

    Returns:
        The configured SEGNN model.
    """

    if isinstance(hidden_irreps, e3nn.Irreps):
        hidden_irreps_units = num_layers * [hidden_irreps]
    else:
        hidden_irreps_units = hidden_irreps

    def _ApplySEGNN(
        graph: SteerableGraphsTuple,
    ) -> jnp.array:

        # embedding
        # NOTE this is not in the original paper but achieves good results
        graph = O3Embedding(
            graph,
            graph.node_attributes,
            hidden_irreps_units[0],
            edge_attributes=graph.edge_attributes,
        )

        # message passing layers
        for n, hrp in enumerate(hidden_irreps_units):
            message_fn, update_fn = SEGNNLayer(
                output_irreps=hrp, layer_num=n, blocks=blocks_per_layer, norm=norm
            )
            graph = SteerableMessagePassing(
                update_fn=update_fn,
                message_fn=message_fn,
                aggregate_messages_fn=jraph.segment_sum,
            )(graph)

        # decoder
        nodes = SEDecoder(
            latent_irreps=hidden_irreps_units[-1],
            output_irreps=output_irreps,
            task=task,
            pool=pool,
        )(graph)

        return nodes.array

    return _ApplySEGNN
