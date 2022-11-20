from functools import partial
from math import floor
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree
from e3nn_jax import Irreps, IrrepsArray, spherical_harmonics
from jraph import segment_mean

from experiments.nbody.nbody_datasets import ChargedDataset, GravityDataset
from segnn import SteerableGraphsTuple


def knn_edges(loc: jnp.array, k: int, n_nodes: int):
    """Returns the k-shortest edges in each batch."""

    @partial(jax.jit, static_argnames=["k"])
    def top_k_linear(loc: jnp.array, k: int) -> jnp.array:
        """Returns the minimum values of a inearized upper triangular matrix."""
        dists = jnp.sum((loc[:, None, :] - loc[None, :, :]) ** 2, axis=-1)
        dists = dists[jnp.triu_indices(loc.shape[0], k=1)]
        return jax.lax.approx_min_k(dists, k)[1]

    def lin_to_triu(k: int, n: int, shift: int) -> Tuple[int, int]:
        """Convert linear to triangular indices."""
        i = n - 2 - int(jnp.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
        j = int(k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2)
        return i + shift, j + shift

    return jnp.array(
        [
            lin_to_triu(int(e), n_nodes, shift=b)
            for b in range(0, loc.shape[0] - 1, n_nodes)
            for e in top_k_linear(loc[b : b + n_nodes], k)
        ]
    ).T


def O3Transform(
    node_features_irreps: Irreps, edge_features_irreps: Irreps, lmax_attributes: int
) -> Callable:
    """
    Build a transformation function that includes (nbody) O3 attributes to a graph.
    """
    attribute_irreps = Irreps.spherical_harmonics(lmax_attributes)

    def _o3_transform(
        graph: SteerableGraphsTuple,
        loc: jnp.ndarray,
        vel: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> SteerableGraphsTuple:

        prod_charges = charges[graph.senders] * charges[graph.receivers]
        rel_pos = loc[graph.senders] - loc[graph.receivers]
        edge_dist = jnp.sqrt(jnp.power(rel_pos, 2).sum(1, keepdims=True))
        # NOTE additional_message_features is the same as edges in this implementation
        edges = IrrepsArray(
            edge_features_irreps,
            jnp.concatenate((edge_dist, prod_charges), axis=-1),
        )

        vel_abs = jnp.sqrt(jnp.power(vel, 2).sum(1, keepdims=True))
        mean_pos = loc.mean(1, keepdims=True)

        nodes = IrrepsArray(
            node_features_irreps,
            jnp.concatenate((loc - mean_pos, vel, vel_abs), axis=1),
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

        return SteerableGraphsTuple(
            nodes=nodes,
            edges=edges,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            senders=graph.senders,
            receivers=graph.receivers,
            n_node=graph.n_node,
            n_edge=graph.n_edge,
            globals=graph.globals,
        )

    return _o3_transform


class NbodyGraphDataloader:
    def __init__(
        self,
        dataset: Union[ChargedDataset, GravityDataset],
        dataset_type: str,
        batch_size: int,
        drop_last: bool = False,
        transform: Optional[Callable] = None,
        neighbours: Optional[int] = 1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self._n_nodes = self.dataset.get_n_nodes()
        self._drop_last = drop_last
        self._dataset_type = dataset_type
        self._neighbours = neighbours

        self._transform = transform

    def __iter__(self):
        i = 0
        if self._dataset_type == "charged":
            edge_indices = self.dataset.get_edges(self.batch_size, self._n_nodes)
            senders, receivers = edge_indices[0], edge_indices[1]
        while i < len(self.dataset):
            cur_batch = min(self.batch_size, len(self.dataset) - i)
            if cur_batch < self.batch_size and self._drop_last:
                break
            if cur_batch < self.batch_size and self._dataset_type == "charged":
                # recompute edges for truncated batch
                edge_indices = self.dataset.get_edges(cur_batch, self._n_nodes)
                senders, receivers = edge_indices[0], edge_indices[1]
            loc, vel, _, q, targets = self.dataset[i : (i + cur_batch)]
            if self._dataset_type == "gravity":
                edge_indices = knn_edges(loc, self._neighbours, self._n_nodes)
                senders, receivers = edge_indices[0], edge_indices[1]
            graph = SteerableGraphsTuple(
                senders=senders,
                receivers=receivers,
                n_node=jnp.array([self._n_nodes] * cur_batch),
                n_edge=jnp.array([len(senders) // cur_batch] * cur_batch),
            )
            graph = self._transform(graph, loc, vel, q)
            # relative shift as target
            if self._dataset_type == "charged":
                targets = targets - loc
            i += cur_batch
            yield graph, targets

    def __len__(self) -> int:
        return floor(self.n_batches) if self._drop_last else round(self.n_batches)

    @property
    def n_batches(self) -> float:
        return len(self.dataset) / self.batch_size
