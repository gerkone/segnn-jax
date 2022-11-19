from typing import Callable, NamedTuple, Optional

import e3nn_jax as e3nn
import jax.numpy as jnp
import jax.tree_util as tree
from jraph import AggregateEdgesToNodesFn, segment_sum


class SteerableGraphsTuple(NamedTuple):
    """Include (steerable) node and edge attributes directly in jraph.GraphsTuple."""

    nodes: Optional[e3nn.IrrepsArray] = None
    edges: Optional[e3nn.IrrepsArray] = None
    receivers: Optional[e3nn.IrrepsArray] = None
    senders: Optional[e3nn.IrrepsArray] = None
    globals: Optional[e3nn.IrrepsArray] = None
    n_node: Optional[jnp.array] = None
    n_edge: Optional[jnp.array] = None
    node_attributes: Optional[e3nn.IrrepsArray] = None
    edge_attributes: Optional[e3nn.IrrepsArray] = None


NodeFeatures = e3nn.IrrepsArray
EdgeFeatures = e3nn.IrrepsArray
Messages = e3nn.IrrepsArray
NodeAttributes = e3nn.IrrepsArray
EdgeAttributes = e3nn.IrrepsArray
AdditionalFeatures = e3nn.IrrepsArray


# edge_attributes, sender_nodes, receiver_nodes, edge_features
MPNNMessageFn = Callable[
    [EdgeAttributes, NodeFeatures, NodeFeatures, EdgeFeatures], Messages
]

# node_attributes, message, node_features
MPNNUpdateFn = Callable[[NodeAttributes, NodeFeatures, Messages], NodeFeatures]


def SteerableMessagePassing(
    message_fn: Optional[MPNNMessageFn],
    update_fn: Optional[MPNNUpdateFn],
    aggregate_messages_fn: AggregateEdgesToNodesFn = segment_sum,
):
    """Returns a method that applies a configured MPNN."""
    # TODO is this needed or can we use the GraphNetwork provided in jraph?
    # judginf from the types this is needed but it's weird
    if message_fn is None or update_fn is None:
        raise ValueError("Must supply both message and update functions.")

    def _ApplyGraphNet(graph: SteerableGraphsTuple) -> SteerableGraphsTuple:
        """Applies a configured MPNN to a SteerableGraphsTuple graph.
        Args:
          graph: a `SteerableGraphsTuple` containing the graph.
        Returns:
          Updated `SteerableGraphsTuple`.
        """
        # pylint: disable=g-long-lambda
        # Equivalent to jnp.sum(n_node), but jittable
        sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]
        if not tree.tree_all(
            tree.tree_map(lambda n: n.shape[0] == sum_n_node, graph.nodes)
        ):
            raise ValueError(
                "All node arrays in nest must contain the same number of nodes."
            )

        sender_nodes = tree.tree_map(lambda n: n[graph.senders], graph.nodes)
        receiver_nodes = tree.tree_map(lambda n: n[graph.receivers], graph.nodes)
        # build message
        msg = message_fn(
            graph.edge_attributes,
            sender_nodes,
            receiver_nodes,
            graph.edges,
        )

        # aggregate messages
        msg = tree.tree_map(
            lambda m: aggregate_messages_fn(m, graph.receivers, sum_n_node), msg
        )

        nodes = update_fn(graph.node_attributes, graph.nodes, msg)

        # pylint: enable=g-long-lambda
        return graph._replace(nodes=nodes)

    return _ApplyGraphNet
