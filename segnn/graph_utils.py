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


def batched_graph_nodes(graph: jraph.GraphsTuple) -> Tuple[jnp.array, int]:
    """Map the (batched) node to the corresponding (batched) graph"""
    n_graphs = graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graphs)
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]
    return (
        jnp.repeat(graph_idx, graph.n_node, axis=0, total_repeat_length=sum_n_node),
        n_graphs,
    )


def pooling(
    nodes: e3nn.IrrepsArray,
    batch: jnp.array,
    n_graphs: int,
    aggregate_fn: Callable = jraph.segment_sum,
) -> e3nn.IrrepsArray:
    """Pools over graph nodes with the specified aggregation.

    Args:
        nodes: input nodes
        batch: batch that maps nodes the corresponding graph
        n_graphs: number of graphs in the batch
        aggregate_fn: function used to update pool over the nodes

    Returns:
        The pooled graph nodes.
    """
    return tree.tree_map(lambda n: aggregate_fn(n, batch, n_graphs), nodes)
