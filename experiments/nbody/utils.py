from typing import Callable, List, Optional, Tuple, Union

import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
import torch
from e3nn_jax import Irreps, IrrepsArray, spherical_harmonics
from jax import random
from jraph import GraphsTuple, segment_mean
from torch_geometric.nn import knn_graph

from experiments.nbody.datasets import ChargedDataset, GravityDataset
from segnn import SteerableGraphsTuple

key = random.PRNGKey(0)


def O3Transform(
    node_features_irreps: Irreps, edge_features_irreps: Irreps, lmax_attributes: int
) -> Callable:
    """
    Build a transformation function that includes (nbody) O3 attributes to a graph.
    """
    attribute_irreps = Irreps.spherical_harmonics(lmax_attributes)

    def _o3_transform(
        st_graph: SteerableGraphsTuple,
        loc: jnp.ndarray,
        vel: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> SteerableGraphsTuple:

        graph = st_graph.graph
        prod_charges = charges[graph.senders] * charges[graph.receivers]
        rel_pos = loc[graph.senders] - loc[graph.receivers]
        edge_dist = jnp.sqrt(jnp.power(rel_pos, 2).sum(1, keepdims=True))

        msg_features = IrrepsArray(
            edge_features_irreps,
            jnp.concatenate((edge_dist, prod_charges), axis=-1),
        )

        vel_abs = jnp.sqrt(jnp.power(vel, 2).sum(1, keepdims=True))
        mean_loc = loc.mean(1, keepdims=True)

        nodes = IrrepsArray(
            node_features_irreps,
            jnp.concatenate((loc - mean_loc, vel, vel_abs), axis=1),
        )

        edge_attributes = spherical_harmonics(
            attribute_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = spherical_harmonics(
            attribute_irreps, vel, normalize=True, normalization="integral"
        )
        # scatter edge attributes
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
        node_attributes = (
            tree.tree_map(
                lambda e: segment_mean(e, graph.receivers, sum_n_node),
                edge_attributes,
            )
            + vel_embedding
        )

        # scalar attribute to 1 by default
        node_attributes.array = node_attributes.array.at[:, 0].set(1.0)

        return SteerableGraphsTuple(
            graph=GraphsTuple(
                nodes=nodes,
                edges=None,
                senders=graph.senders,
                receivers=graph.receivers,
                n_node=graph.n_node,
                n_edge=graph.n_edge,
                globals=graph.globals,
            ),
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            additional_message_features=msg_features,
        )

    return _o3_transform


def GraphTransform(
    transform: Callable, neighbours: Optional[int] = 0, relative_target: bool = False
) -> Callable:
    def _to_steerable_graph(
        dataset: Union[ChargedDataset, GravityDataset], data: List
    ) -> Tuple[SteerableGraphsTuple, jnp.ndarray]:

        loc, vel, _, q, targets = data

        n_nodes = dataset.get_n_nodes()
        batch_size = int(len(data[0]) / n_nodes)

        if dataset.data_type == "charged":
            edge_indices = dataset.get_edges(batch_size, n_nodes)
            senders, receivers = edge_indices[0], edge_indices[1]
        if dataset.data_type == "gravity":
            batch = torch.arange(0, batch_size)
            batch = batch.repeat_interleave(n_nodes).long()
            edge_indices = knn_graph(torch.from_numpy(np.array(loc)), neighbours, batch)
            senders = jnp.array(edge_indices[0])
            receivers = jnp.array(edge_indices[1])

        st_graph = SteerableGraphsTuple(
            graph=GraphsTuple(
                nodes=None,
                edges=None,
                senders=senders,
                receivers=receivers,
                n_node=jnp.array([n_nodes] * batch_size),
                n_edge=jnp.array([len(senders) // batch_size] * batch_size),
                globals=None,
            )
        )
        st_graph = transform(st_graph, loc, vel, q)
        # relative shift as target
        if relative_target:
            targets = targets - loc

        return st_graph, targets

    return _to_steerable_graph


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return jnp.vstack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)
