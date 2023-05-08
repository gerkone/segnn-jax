from typing import Any, Callable, List, Optional, Union

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp
import jraph
from jax.tree_util import Partial

from .blocks import O3Layer, O3TensorProductGate
from .graph_utils import SteerableGraphsTuple, pooling


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
        graph = st_graph.graph
        nodes = O3Layer(
            embed_irreps,
            name="embedding_nodes",
        )(graph.nodes, st_graph.node_attributes)
        st_graph = st_graph._replace(graph=graph._replace(nodes=nodes))

        # NOTE edge embedding is not in the original paper but can get good results
        if embed_edges:
            additional_message_features = O3Layer(
                embed_irreps,
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


def O3Decoder(
    latent_irreps: e3nn.Irreps,
    output_irreps: e3nn.Irreps,
    blocks: int = 1,
    task: str = "graph",
    pool: Optional[str] = "avg",
):
    """Steerable pooler and decoder.

    Args:
        latent_irreps: Representation from the previous block
        output_irreps: Output representation
        blocks: Number of tensor product blocks in the decoder
        task: Specifies where the output is located. Either 'graph' or 'node'

    Returns:
        Decoded latent feature space to output space.
    """

    assert task in ["node", "graph"], f"Unknown task {task}"
    assert pool in ["avg", "sum", "none", None], f"Unknown pooling '{pool}'"

    def _decoder(st_graph: SteerableGraphsTuple):
        nodes = st_graph.graph.nodes
        # pre pool block
        for i in range(blocks):
            nodes = O3TensorProductGate(latent_irreps, name=f"prepool_{i}")(
                nodes, st_graph.node_attributes
            )

        if task == "node":
            nodes = O3Layer(output_irreps, name="output")(
                nodes, st_graph.node_attributes
            )

        if task == "graph":
            # pool over graph
            pooled_irreps = (latent_irreps.num_irreps * output_irreps).regroup()
            nodes = O3Layer(pooled_irreps, name=f"prepool_{blocks}")(
                nodes, st_graph.node_attributes
            )

            # pooling layer
            if pool == "avg":
                pool_fn = jraph.segment_mean
            if pool == "sum":
                pool_fn = jraph.segment_sum

            nodes = pooling(st_graph.graph._replace(nodes=nodes), aggregate_fn=pool_fn)

            # post pool mlp (not steerable)
            for i in range(blocks):
                nodes = O3TensorProductGate(pooled_irreps, name=f"postpool_{i}")(nodes)
            nodes = O3Layer(output_irreps, name="output")(nodes)

        return nodes

    return _decoder


class SEGNNLayer(hk.Module):
    """
    Steerable E(3) equivariant layer.

    Applies a message passing step (GN) with equivariant message and update functions.
    """

    def __init__(
        self,
        output_irreps: e3nn.Irreps,
        layer_num: int,
        blocks: int = 2,
        norm: Optional[str] = None,
        aggregate_fn: Optional[Callable] = jraph.segment_sum,
    ):
        """
        Initialize the layer.

        Args:
            output_irreps: Layer output representation
            layer_num: Numbering of the layer
            blocks: Number of tensor product blocks in the layer
            norm: Normalization type. Either be None, 'instance' or 'batch'
            aggregate_fn: Message aggregation function. Defaults to sum.
        """
        super().__init__(f"layer_{layer_num}")
        assert norm in ["batch", "instance", "none", None], f"Unknown norm '{norm}'"
        self._output_irreps = output_irreps
        self._blocks = blocks
        self._norm = norm
        self._aggregate_fn = aggregate_fn

    def _message(
        self,
        edge_attribute: e3nn.IrrepsArray,
        additional_message_features: e3nn.IrrepsArray,
        edge_features: Any,
        incoming: e3nn.IrrepsArray,
        outgoing: e3nn.IrrepsArray,
        globals_: Any,
    ) -> e3nn.IrrepsArray:
        """Steerable equivariant message function."""
        _ = globals_
        _ = edge_features
        # create messages
        msg = e3nn.concatenate([incoming, outgoing], axis=-1)
        if additional_message_features is not None:
            msg = e3nn.concatenate([msg, additional_message_features], axis=-1)
        # message mlp (phi_m in the paper) steered by edge attributeibutes
        for i in range(self._blocks):
            msg = O3TensorProductGate(self._output_irreps, name=f"tp_{i}")(
                msg, edge_attribute
            )
        # NOTE: original implementation only applied batch norm to messages
        if self._norm == "batch":
            msg = e3nn.haiku.BatchNorm(irreps=self._output_irreps)(msg)
        return msg

    def _update(
        self,
        node_attribute: e3nn.IrrepsArray,
        nodes: e3nn.IrrepsArray,
        senders: Any,
        msg: e3nn.IrrepsArray,
        globals_: Any,
    ) -> e3nn.IrrepsArray:
        """Steerable equivariant update function."""
        _ = senders
        _ = globals_
        x = e3nn.concatenate([nodes, msg], axis=-1)
        # update mlp (phi_f in the paper) steered by node attributeibutes
        for i in range(self._blocks - 1):
            x = O3TensorProductGate(self._output_irreps, name=f"tp_{i}")(
                x, node_attribute
            )
        # last update layer without activation
        update = O3Layer(self._output_irreps, name=f"tp_{self._blocks - 1}")(
            x, node_attribute
        )
        # residual connection
        nodes += update
        # message norm
        if self._norm in ["batch", "instance"]:
            nodes = e3nn.haiku.BatchNorm(
                irreps=self._output_irreps,
                instance=(self._norm == "instance"),
            )(nodes)
        return nodes

    def __call__(self, st_graph: SteerableGraphsTuple) -> SteerableGraphsTuple:
        """Perform a message passing step.

        Args:
            st_graph: Input graph

        Returns:
            The updated graph
        """
        # NOTE node_attributes, edge_attributes and additional_message_features
        #  are never updated within the message passing layers
        return st_graph._replace(
            graph=jraph.GraphNetwork(
                update_node_fn=Partial(self._update, st_graph.node_attributes),
                update_edge_fn=Partial(
                    self._message,
                    st_graph.edge_attributes,
                    st_graph.additional_message_features,
                ),
                aggregate_edges_for_nodes_fn=self._aggregate_fn,
            )(st_graph.graph)
        )


class SEGNN(hk.Module):
    """Steerable E(3) equivariant network.

    Original paper https://arxiv.org/abs/2110.02905.
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
        """
        Initialize the network.

        Args:
            hidden_irreps: Feature representation in the hidden layers
            output_irreps: Output representation.
            num_layers: Number of message passing layers
            norm: Normalization type. Either None, 'instance' or 'batch'
            pool: Pooling mode (only for graph-wise tasks)
            task: Specifies where the output is located. Either 'graph' or 'node'
            blocks_per_layer: Number of tensor product blocks in each message passing
            embed_msg_features: Set to true to also embed edges/message passing features
        """
        super().__init__()

        if isinstance(hidden_irreps, e3nn.Irreps):
            self._hidden_irreps_units = num_layers * [hidden_irreps]
        else:
            self._hidden_irreps_units = hidden_irreps

        self._embed_msg_features = embed_msg_features
        self._norm = norm
        self._blocks_per_layer = blocks_per_layer

        self._embedding = O3Embedding(
            self._hidden_irreps_units[0],
            embed_edges=self._embed_msg_features,
        )

        self._decoder = O3Decoder(
            latent_irreps=self._hidden_irreps_units[-1],
            output_irreps=output_irreps,
            task=task,
            pool=pool,
        )

    def __call__(self, st_graph: SteerableGraphsTuple) -> jnp.array:
        # node (and edge) embedding
        st_graph = self._embedding(st_graph)

        # message passing
        for n, hrp in enumerate(self._hidden_irreps_units):
            st_graph = SEGNNLayer(output_irreps=hrp, layer_num=n, norm=self._norm)(
                st_graph
            )

        # decoder/pooler
        nodes = self._decoder(st_graph)

        return jnp.squeeze(nodes.array)
