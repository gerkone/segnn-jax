from typing import Callable, Tuple

import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from segnn_jax import SteerableGraphsTuple

from .dataset import QM9


def QM9GraphTransform(
    args,
    max_batch_nodes: int,
    max_batch_edges: int,
    train_trn: Callable,
) -> Callable:
    """
    Build a function that converts torch DataBatch into SteerableGraphsTuple.

    Mostly a quick fix out of lazyness. Rewriting QM9 in jax is not trivial.
    """
    attribute_irreps = e3nn.Irreps.spherical_harmonics(args.lmax_attributes)

    def _to_steerable_graph(
        data: Data, training: bool = True
    ) -> Tuple[SteerableGraphsTuple, jnp.array]:
        ptr = jnp.array(data.ptr)
        senders = jnp.array(data.edge_index[0])
        receivers = jnp.array(data.edge_index[1])
        graph = jraph.GraphsTuple(
            nodes=e3nn.IrrepsArray(args.node_irreps, jnp.array(data.x)),
            edges=None,
            senders=senders,
            receivers=receivers,
            n_node=jnp.diff(ptr),
            n_edge=jnp.diff(jnp.sum(senders[:, jnp.newaxis] < ptr, axis=0)),
            globals=None,
        )
        # pad for jax static shapes
        node_attr_pad = ((0, max_batch_nodes - jnp.sum(graph.n_node) + 1), (0, 0))
        edge_attr_pad = ((0, max_batch_edges - jnp.sum(graph.n_edge) + 1), (0, 0))
        graph = jraph.pad_with_graphs(
            graph,
            n_node=max_batch_nodes + 1,
            n_edge=max_batch_edges + 1,
            n_graph=graph.n_node.shape[0] + 1,
        )

        node_attributes = e3nn.IrrepsArray(
            attribute_irreps, jnp.pad(jnp.array(data.node_attr), node_attr_pad)
        )
        # scalar attribute to 1 by default
        node_attributes = e3nn.IrrepsArray(
            node_attributes.irreps, node_attributes.array.at[:, 0].set(1.0)
        )

        additional_message_features = e3nn.IrrepsArray(
            args.additional_message_irreps,
            jnp.pad(jnp.array(data.additional_message_features), edge_attr_pad),
        )
        edge_attributes = e3nn.IrrepsArray(
            attribute_irreps, jnp.pad(jnp.array(data.edge_attr), edge_attr_pad)
        )

        st_graph = SteerableGraphsTuple(
            graph=graph,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            additional_message_features=additional_message_features,
        )

        # pad targets
        target = jnp.array(data.y)
        if args.task == "node":
            target = jnp.pad(target, [(0, max_batch_nodes - target.shape[0] - 1)])
        if args.task == "graph":
            target = jnp.append(target, 0)

        # normalize targets
        if training and train_trn is not None:
            target = train_trn(target)

        return st_graph, target

    return _to_steerable_graph


def setup_qm9_data(
    args,
) -> Tuple[DataLoader, DataLoader, DataLoader, Callable, Callable]:
    dataset_train = QM9(
        "datasets",
        args.target,
        args.radius,
        partition="train",
        lmax_attr=args.lmax_attributes,
        feature_type=args.feature_type,
    )
    dataset_val = QM9(
        "datasets",
        args.target,
        args.radius,
        partition="valid",
        lmax_attr=args.lmax_attributes,
        feature_type=args.feature_type,
    )
    dataset_test = QM9(
        "datasets",
        args.target,
        args.radius,
        partition="test",
        lmax_attr=args.lmax_attributes,
        feature_type=args.feature_type,
    )

    # 0.8 (un)safety factor for rejitting
    max_batch_nodes = int(0.8 * sum(dataset_test.top_n_nodes(args.batch_size)))
    max_batch_edges = int(0.8 * sum(dataset_test.top_n_edges(args.batch_size)))

    target_mean, target_mad = dataset_train.calc_stats()

    def remove_offsets(t):
        return (t - target_mean) / target_mad

    # not great and very slow due to huge padding
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    to_graphs_tuple = QM9GraphTransform(
        args,
        max_batch_nodes=max_batch_nodes,
        max_batch_edges=max_batch_edges,
        train_trn=remove_offsets,
    )

    def add_offsets(p):
        return p * target_mad + target_mean

    return loader_train, loader_val, loader_test, to_graphs_tuple, add_offsets
