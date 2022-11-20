from typing import Callable, List, Optional, Tuple, Union

import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph

from .blocks import O3Embedding, O3TensorProduct, O3TensorProductGate
from .graphs import SteerableGraphsTuple, SteerableMessagePassing


def SEDecoder(
    latent_irreps: e3nn.Irreps,
    output_irreps: e3nn.Irreps,
    blocks: int = 1,
    task: str = "graph",
):
    r"""Steerable E(3) Decoder
    Args:
        latent_irreps
        output_irreps:
        blocks:
        task:
    """

    def _ApplySEDecoder(graph: SteerableGraphsTuple):
        nodes = graph.nodes
        if task == "graph":
            raise NotImplementedError("Graph based tasks not yet implemented.")
        elif task == "node":
            # label nodes directly
            pooled_irreps = latent_irreps

        # output block
        for _ in range(blocks):
            nodes = e3nn.Linear(pooled_irreps, biases=True)(nodes)
            nodes = e3nn.gate(nodes)

        return e3nn.Linear(output_irreps, biases=True)(nodes)

    return _ApplySEDecoder


def SEGNNLayer(
    output_irreps: e3nn.Irreps, blocks: int = 2, norm: Optional[str] = None
) -> Tuple[Callable, Callable]:
    r"""Steerable E(3) equivariant layer
    Args:
        output_irreps:
        blocks:
        norm:
    """

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
        for _ in range(blocks):
            msg = O3TensorProductGate(msg, edge_attribute, output_irreps)
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
        for _ in range(blocks - 1):
            x = O3TensorProductGate(x, node_attribute, output_irreps)
        # last update layer without activation
        update = O3TensorProduct(x, node_attribute, output_irreps)
        # residual connection
        nodes += update
        if norm == "batch":
            nodes = e3nn.BatchNorm(irreps=output_irreps)(nodes)
        elif norm == "instance":
            raise NotImplementedError("Instance norm not yet implemented")
        return nodes

    return _message, _update


def SEGNN(
    hidden_irreps: Union[List[e3nn.Irreps], e3nn.Irreps],
    output_irreps: e3nn.Irreps,
    num_layers: Optional[int],
    norm: Optional[str] = None,
    task: Optional[str] = "graph",
    blocks_per_layer: int = 2,
):
    r"""Steerable E(3) equivariant network
    Args:
        hidden_irreps:
        output_irreps:
        num_layers:
        norm:
        pool:
        task:
        blocks_per_layer:
    """
    if isinstance(hidden_irreps, e3nn.Irreps):
        hidden_irreps_units = num_layers * [hidden_irreps]
    else:
        hidden_irreps_units = hidden_irreps

    def _ApplySEGNN(
        # TODO should force the data be IrrepArrays everywhere or better to pass irreps around?
        graph: SteerableGraphsTuple,
    ) -> jnp.array:
        # embedding
        graph = O3Embedding(graph, graph.node_attributes, hidden_irreps_units[0])

        # message passing layers
        for hrp in hidden_irreps_units:
            message_fn, update_fn = SEGNNLayer(
                output_irreps=hrp, blocks=blocks_per_layer, norm=norm
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
        )(graph)

        return nodes.array

    return _ApplySEGNN
