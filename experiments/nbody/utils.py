from typing import Callable, List, Optional, Tuple

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
import torch
from jraph import GraphsTuple, segment_mean
from torch.utils.data import DataLoader
from torch_geometric.nn import knn_graph

from segnn_jax import SteerableGraphsTuple

from .datasets import ChargedDataset, GravityDataset


def O3Transform(
    node_features_irreps: e3nn.Irreps,
    edge_features_irreps: e3nn.Irreps,
    lmax_attributes: int,
) -> Callable:
    """
    Build a transformation function that includes (nbody) O3 attributes to a graph.
    """
    attribute_irreps = e3nn.Irreps.spherical_harmonics(lmax_attributes)

    @jax.jit
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

        msg_features = e3nn.IrrepsArray(
            edge_features_irreps,
            jnp.concatenate((edge_dist, prod_charges), axis=-1),
        )

        vel_abs = jnp.sqrt(jnp.power(vel, 2).sum(1, keepdims=True))
        mean_loc = loc.mean(1, keepdims=True)

        nodes = e3nn.IrrepsArray(
            node_features_irreps,
            jnp.concatenate((loc - mean_loc, vel, vel_abs), axis=-1),
        )

        edge_attributes = e3nn.spherical_harmonics(
            attribute_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = e3nn.spherical_harmonics(
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
        node_attributes = e3nn.IrrepsArray(
            node_attributes.irreps, node_attributes.array.at[:, 0].set(1.0)
        )

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


def NbodyGraphTransform(
    transform: Callable,
    dataset_name: str,
    n_nodes: int,
    batch_size: int,
    neighbours: Optional[int] = 6,
    relative_target: bool = False,
) -> Callable:
    """
    Build a function that converts torch DataBatch into SteerableGraphsTuple.
    """

    if dataset_name == "charged":
        # charged system is a connected graph
        full_edge_indices = jnp.array(
            [
                (i + n_nodes * b, j + n_nodes * b)
                for b in range(batch_size)
                for i in range(n_nodes)
                for j in range(n_nodes)
                if i != j
            ]
        ).T

    def _to_steerable_graph(
        data: List, training: bool = True
    ) -> Tuple[SteerableGraphsTuple, jnp.ndarray]:
        _ = training
        loc, vel, _, q, targets = data

        cur_batch = int(loc.shape[0] / n_nodes)

        if dataset_name == "charged":
            edge_indices = full_edge_indices[:, : n_nodes * (n_nodes - 1) * cur_batch]
            senders, receivers = edge_indices[0], edge_indices[1]
        if dataset_name == "gravity":
            batch = torch.arange(0, cur_batch)
            batch = batch.repeat_interleave(n_nodes).long()
            edge_indices = knn_graph(torch.from_numpy(np.array(loc)), neighbours, batch)
            # switched by default
            senders, receivers = jnp.array(edge_indices[0]), jnp.array(edge_indices[1])

        st_graph = SteerableGraphsTuple(
            graph=GraphsTuple(
                nodes=None,
                edges=None,
                senders=senders,
                receivers=receivers,
                n_node=jnp.array([n_nodes] * cur_batch),
                n_edge=jnp.array([len(senders) // cur_batch] * cur_batch),
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


def setup_nbody_data(
    args,
) -> Tuple[DataLoader, DataLoader, DataLoader, Callable, Callable]:
    if args.dataset == "charged":
        dataset_train = ChargedDataset(
            partition="train",
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
            n_bodies=args.n_bodies,
        )
        dataset_val = ChargedDataset(
            partition="val",
            dataset_name=args.dataset_name,
            n_bodies=args.n_bodies,
        )
        dataset_test = ChargedDataset(
            partition="test",
            dataset_name=args.dataset_name,
            n_bodies=args.n_bodies,
        )

    if args.dataset == "gravity":
        dataset_train = GravityDataset(
            partition="train",
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
            neighbours=args.neighbours,
            target=args.target,
            n_bodies=args.n_bodies,
        )
        dataset_val = GravityDataset(
            partition="val",
            dataset_name=args.dataset_name,
            neighbours=args.neighbours,
            target=args.target,
            n_bodies=args.n_bodies,
        )
        dataset_test = GravityDataset(
            partition="test",
            dataset_name=args.dataset_name,
            neighbours=args.neighbours,
            target=args.target,
            n_bodies=args.n_bodies,
        )

    o3_transform = O3Transform(
        args.node_irreps, args.additional_message_irreps, args.lmax_attributes
    )
    graph_transform = NbodyGraphTransform(
        transform=o3_transform,
        n_nodes=args.n_bodies,
        batch_size=args.batch_size,
        neighbours=args.neighbours,
        relative_target=(args.target == "pos"),
        dataset_name=args.dataset,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=numpy_collate,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )

    return loader_train, loader_val, loader_test, graph_transform, None
