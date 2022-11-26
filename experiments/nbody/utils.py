from math import floor
from typing import Callable, Optional, Union

import jax.numpy as jnp
import jax.tree_util as tree
from e3nn_jax import Irreps, IrrepsArray, spherical_harmonics
from jraph import GraphsTuple, segment_mean

from experiments.nbody.datasets import ChargedDataset, GravityDataset
from segnn import SteerableGraphsTuple


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


class NbodyGraphDataloader:
    """Dataloader for the N-body datasets, directly handles graph features and attributes."""

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
        elif self._dataset_type == "gravity":
            from pcdiff import knn_graph
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
                edge_indices = knn_graph(loc, self._neighbours)
                senders, receivers = jnp.array(edge_indices[0]), jnp.array(
                    edge_indices[1]
                )

            st_graph = SteerableGraphsTuple(
                graph=GraphsTuple(
                    nodes=None,
                    edges=None,
                    senders=senders,
                    receivers=receivers,
                    n_node=jnp.array([self._n_nodes] * cur_batch),
                    n_edge=jnp.array([len(senders) // cur_batch] * cur_batch),
                    globals=None,
                )
            )
            st_graph = self._transform(st_graph, loc, vel, q)
            # relative shift as target
            # if self._dataset_type == "charged":
            targets = targets - loc
            i += cur_batch
            yield st_graph, targets

    def __len__(self) -> int:
        return floor(self.n_batches) if self._drop_last else round(self.n_batches)

    @property
    def n_batches(self) -> float:
        return len(self.dataset) / self.batch_size


def setup_nbody_data(args):
    if args.dataset == "charged":
        dataset_train = ChargedDataset(
            partition="train",
            dataset_name=args.dataset_partition,
            max_samples=args.max_samples,
            n_bodies=args.n_bodies,
        )
        dataset_val = ChargedDataset(
            partition="val", dataset_name=args.dataset_partition, n_bodies=args.n_bodies
        )
        dataset_test = ChargedDataset(
            partition="test",
            dataset_name=args.dataset_partition,
            n_bodies=args.n_bodies,
        )

    if args.dataset == "gravity":
        dataset_train = GravityDataset(
            partition="train",
            dataset_name=args.dataset_partition,
            max_samples=args.max_samples,
            neighbours=args.neighbours,
            target=args.target,
            n_bodies=args.n_bodies,
        )
        dataset_val = GravityDataset(
            partition="val",
            dataset_name=args.dataset_partition,
            neighbours=args.neighbours,
            target=args.target,
            n_bodies=args.n_bodies,
        )
        dataset_test = GravityDataset(
            partition="test",
            dataset_name=args.dataset_partition,
            neighbours=args.neighbours,
            target=args.target,
            n_bodies=args.n_bodies,
        )

    o3_transform = O3Transform(args.node_irreps, args.edge_irreps, args.lmax_attributes)

    loader_train = NbodyGraphDataloader(
        dataset_train,
        args.dataset,
        args.batch_size,
        drop_last=False,
        transform=o3_transform,
        neighbours=args.neighbours if args.dataset == "gravity" else None,
    )
    loader_val = NbodyGraphDataloader(
        dataset_val,
        args.dataset,
        args.batch_size,
        drop_last=True,
        transform=o3_transform,
        neighbours=args.neighbours if args.dataset == "gravity" else None,
    )
    loader_test = NbodyGraphDataloader(
        dataset_test,
        args.dataset,
        args.batch_size,
        drop_last=True,
        transform=o3_transform,
        neighbours=args.neighbours if args.dataset == "gravity" else None,
    )

    return loader_train, loader_val, loader_test
