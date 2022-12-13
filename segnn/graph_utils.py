from typing import Callable, NamedTuple, Optional, Tuple

import e3nn_jax as e3nn
import jax.numpy as jnp
import jax.tree_util as tree
import jraph


class SteerableGraphsTuple(NamedTuple):
    """Pack (steerable) node and edge attributes with jraph.GraphsTuple."""

    graph: jraph.GraphsTuple
    node_attributes: Optional[e3nn.IrrepsArray] = None
    edge_attributes: Optional[e3nn.IrrepsArray] = None
    # NOTE must put additional_message_features in a separate field otherwise
    #  it would get updated by jraph.GraphNetwork (which goes against one of
    #  the core ideas of the paper).
    additional_message_features: Optional[e3nn.IrrepsArray] = None


def pooling(
    graph: jraph.GraphsTuple,
    aggregate_fn: Callable = jraph.segment_sum,
) -> e3nn.IrrepsArray:
    """Pools over graph nodes with the specified aggregation.

    Args:
        graph: Input graph
        aggregate_fn: function used to update pool over the nodes

    Returns:
        The pooled graph nodes.
    """
    n_graphs = graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graphs)
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]
    batch = jnp.repeat(graph_idx, graph.n_node, axis=0, total_repeat_length=sum_n_node)
    return e3nn.IrrepsArray(
        graph.nodes.irreps, aggregate_fn(graph.nodes.array, batch, n_graphs)
    )
