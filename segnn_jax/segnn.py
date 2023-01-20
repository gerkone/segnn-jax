from typing import Any, Callable, List, Optional, Tuple, Union

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp
import jraph
from jax.tree_util import Partial

from .blocks import O3Embedding, O3TensorProduct, O3TensorProductGate
from .graph_utils import SteerableGraphsTuple, pooling


def SEDecoder(
    latent_irreps: e3nn.Irreps,
    output_irreps: e3nn.Irreps,
    blocks: int = 1,
    task: str = "graph",
    pool: Optional[str] = "avg",
):
    """Steerable E(3) pooler and decoder.

    Args:
        latent_irreps: Representation from the previous block
        output_irreps: Output representation
        blocks: Number of tensor product blocks in the decoder
        task: Specifies where the output is located. Either 'graph' or 'node'

    Returns:
        Decoded latent feature space to output space.
    """

    assert task in ["node", "graph"], f"Unknown task {task}"
    assert pool in ["avg", "sum", "none", None], f"Unknown pooling {pool}"

    def _ApplySEDecoder(st_graph: SteerableGraphsTuple):
        nodes = st_graph.graph.nodes
        # pre pool block
        for i in range(blocks):
            nodes = O3TensorProductGate(
                nodes, st_graph.node_attributes, latent_irreps, name=f"prepool_{i}"
            )

        if task == "node":
            nodes = O3TensorProduct(
                nodes, st_graph.node_attributes, output_irreps, name="output"
            )

        if task == "graph":
            # pool over graph
            pooled_irreps = (latent_irreps.num_irreps * output_irreps).regroup()
            nodes = O3TensorProduct(
                nodes, st_graph.node_attributes, pooled_irreps, name=f"prepool_{blocks}"
            )

            # pooling layer
            if pool == "avg":
                pool_fn = jraph.segment_mean
            if pool == "sum":
                pool_fn = jraph.segment_sum

            nodes = pooling(st_graph.graph._replace(nodes=nodes), aggregate_fn=pool_fn)

            # post pool mlp (not steerable)
            for i in range(blocks):
                nodes = O3TensorProductGate(
                    nodes, None, pooled_irreps, name=f"postpool_{i}"
                )
            nodes = O3TensorProduct(nodes, None, output_irreps, name="output")

        return nodes

    return _ApplySEDecoder


def SEGNNLayer(
    output_irreps: e3nn.Irreps,
    layer_num: int,
    blocks: int = 2,
    norm: Optional[str] = None,
) -> Tuple[Callable, Callable]:
    """Steerable E(3) equivariant layer
    Args:
        output_irreps: Layer output representation
        layer_num: Numbering of the layer
        blocks: Number of tensor product blocks in the layer
        norm: Normalization type. Either be None, 'instance' or 'batch'

    Returns:
        Two function compatible with the jraph networks, message and node
        update mlps respectively.
    """

    assert norm in ["batch", "instance", "none", None], f"Unknown normalization {norm}"

    def _message(
        edge_attribute: e3nn.IrrepsArray,
        additional_message_features: e3nn.IrrepsArray,
        edge_features: Any,
        incoming: e3nn.IrrepsArray,
        outgoing: e3nn.IrrepsArray,
        globals_: Any,
    ) -> e3nn.IrrepsArray:
        _ = globals_
        _ = edge_features
        # create messages
        msg = e3nn.concatenate([incoming, outgoing], axis=-1)
        if additional_message_features is not None:
            msg = e3nn.concatenate([msg, additional_message_features], axis=-1)
        # message mlp (phi_m in the paper) steered by edge attributeibutes
        for i in range(blocks):
            msg = O3TensorProductGate(
                msg, edge_attribute, output_irreps, name=f"message_{i}_{layer_num}"
            )
        # NOTE: original implementation only applied batch norm to messages
        if norm == "batch":
            msg = e3nn.BatchNorm(irreps=output_irreps)(msg)
        return msg

    def _update(
        node_attribute: e3nn.IrrepsArray,
        nodes: e3nn.IrrepsArray,
        senders: Any,
        msg: e3nn.IrrepsArray,
        globals_: Any,
    ) -> e3nn.IrrepsArray:
        _ = senders
        _ = globals_
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
        # message norm
        if norm in ["batch", "instance"]:
            nodes = e3nn.BatchNorm(irreps=output_irreps, instance=(norm == "instance"))(
                nodes
            )
        return nodes

    return _message, _update


class SEGNN(hk.Module):
    """Steerable E(3) equivariant network.

    Original paper https://arxiv.org/abs/2110.02905.

    Attributes:
        hidden_irreps: Feature representation in the hidden layers
        output_irreps: Output representation.
        num_layers: Number of message passing layers
        norm: Normalization type. Either be None, 'instance' or 'batch'
        pool: Pooling mode (only for graph-wise tasks)
        task: Specifies where the output is located. Either 'graph' or 'node'
        blocks_per_layer: Number of tensor product blocks in each message passing
        embed_msg_features: Set to true to also embed edges/message passing features
    """

    def __init__(
        self,
        hidden_irreps: Union[List[e3nn.Irreps], e3nn.Irreps],
        output_irreps: e3nn.Irreps,
        num_layers: int,
        norm: Optional[str] = None,
        pool: Optional[str] = "avg",
        task: Optional[str] = "graph",
        blocks_per_layer: int = 2,
        embed_msg_features: bool = False,
    ):
        super(SEGNN, self).__init__()  # noqa # pylint: disable=R1725

        if isinstance(hidden_irreps, e3nn.Irreps):
            self._hidden_irreps_units = num_layers * [hidden_irreps]
        else:
            self._hidden_irreps_units = hidden_irreps

        self._embed_msg_features = embed_msg_features
        self._norm = norm
        self._blocks_per_layer = blocks_per_layer

        self._decoder = SEDecoder(
            latent_irreps=self._hidden_irreps_units[-1],
            output_irreps=output_irreps,
            task=task,
            pool=pool,
        )

    def _propagate(
        self, st_graph: SteerableGraphsTuple, irreps: e3nn.Irreps, layer_num: int
    ) -> SteerableGraphsTuple:
        """Perform a message passing step.

        Args:
            st_graph: Input graph
            irreps: Irreps in the hidden layer
            layer_num: Numbering of the layer

        Returns:
            The updated graph
        """
        message_fn, update_fn = SEGNNLayer(
            output_irreps=irreps,
            layer_num=layer_num,
            blocks=self._blocks_per_layer,
            norm=self._norm,
        )
        # NOTE node_attributes, edge_attributes and additional_message_features
        #  are never updated within the message passing layers
        return st_graph._replace(
            graph=jraph.GraphNetwork(
                update_node_fn=Partial(update_fn, st_graph.node_attributes),
                update_edge_fn=Partial(
                    message_fn,
                    st_graph.edge_attributes,
                    st_graph.additional_message_features,
                ),
                aggregate_edges_for_nodes_fn=jraph.segment_sum,
            )(st_graph.graph)
        )

    def __call__(self, st_graph: SteerableGraphsTuple) -> jnp.array:
        # embedding
        # NOTE edge embedding is not in the original paper but can get good results
        st_graph = O3Embedding(
            st_graph,
            self._hidden_irreps_units[0],
            embed_msg_features=self._embed_msg_features,
        )

        # message passing layers
        for n, hrp in enumerate(self._hidden_irreps_units):
            st_graph = self._propagate(st_graph, irreps=hrp, layer_num=n)

        # decoder
        nodes = self._decoder(st_graph)

        return jnp.squeeze(nodes.array)
