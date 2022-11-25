import argparse
import time
from typing import Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from e3nn_jax import Irreps

import wandb
from experiments.nbody.datasets import ChargedDataset, GravityDataset
from experiments.nbody.utils import NbodyGraphDataloader, O3Transform
from segnn import SEGNN, SteerableGraphsTuple, weight_balanced_irreps

key = jax.random.PRNGKey(0)


def train(segnn: hk.Transformed, dataset_train, dataset_val, dataset_test, args):
    # load data
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

    print("Jitting...")

    params, segnn_state = segnn.init(key, next(iter(loader_train))[0])
    opt_init, opt_update = optax.adamw(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )

    @jax.jit
    def predict(
        params: hk.Params, state: hk.State, graph: SteerableGraphsTuple
    ) -> Tuple[jnp.ndarray, hk.State]:
        return segnn.apply(params, state, graph)

    @jax.jit
    def mse(
        params: hk.Params,
        state: hk.State,
        graph: SteerableGraphsTuple,
        target: jnp.ndarray,
    ) -> float:
        pred, _ = predict(params, state, graph)
        return (jnp.square(pred - target)).mean()

    @jax.jit
    def update(
        params: hk.Params,
        state: hk.State,
        graph: SteerableGraphsTuple,
        target: jnp.ndarray,
        opt_state,
    ) -> Tuple[float, hk.Params, Any]:
        loss, grads = jax.value_and_grad(mse)(params, state, graph, target)
        updates, opt_state = opt_update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), opt_state

    opt_state = opt_init(params)
    avg_time = []

    for e in range(args.epochs):
        train_loss = 0
        train_start = time.perf_counter_ns()
        for graph, target in loader_train:
            loss, params, opt_state = update(
                params, segnn_state, graph, target, opt_state
            )
            train_loss += loss
        train_time = (
            (time.perf_counter_ns() - train_start) / 1e6 / loader_train.n_batches
        )
        train_loss /= loader_train.n_batches
        if args.wandb:
            wandb.log({"train_loss": train_loss, "update_time": train_time})
        print(
            "[Epoch {:>4}] training loss {:.6f}, update time {:.3f}ms".format(
                e + 1, train_loss, train_time
            ),
            end="",
        )
        if e % args.val_freq == 0:
            val_loss = 0
            eval_start = time.perf_counter_ns()
            for graph, target in loader_val:
                val_loss += jax.lax.stop_gradient(
                    mse(params, segnn_state, graph, target)
                )
            eval_time = (
                (time.perf_counter_ns() - eval_start) / 1e6 / loader_val.n_batches
            )
            avg_time.append(eval_time)
            val_loss /= loader_val.n_batches
            if args.wandb:
                wandb.log({"val_loss": val_loss, "eval_time": eval_time})
            print(
                " - validation loss {:.6f}, eval time {:.3f}ms ({} graph batch)".format(
                    val_loss, eval_time, args.batch_size
                ),
                end="",
            )
        print()

    test_loss = 0
    for graph, target in loader_test:
        test_loss += jax.lax.stop_gradient(mse(params, segnn_state, graph, target))
    test_loss /= loader_test.n_batches
    avg_time = sum(avg_time) / len(avg_time)
    if args.wandb:
        wandb.log({"test_loss": test_loss, "avg_eval_time": avg_time})
    print(
        "Training done. Test loss {:.6f} - "
        "eval time {:.3f}ms (average, {} graph batch)".format(
            test_loss, avg_time, args.batch_size
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size.",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-8, help="Weight decay")
    parser.add_argument(
        "--dataset", type=str, default="charged", help="Dataset [charged, gravity]"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Maximum number of samples in nbody dataset",
    )
    parser.add_argument(
        "--n-bodies",
        type=int,
        default=5,
        help="Number of bodies in the dataset",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=10,
        help="Evaluation frequency (number of epochs)",
    )

    # gravity parameters
    parser.add_argument(
        "--target", type=str, default="pos", help="Target (gravity only) [pos, force]"
    )
    parser.add_argument(
        "--neighbours",
        type=int,
        default=20,
        help="Number of connected nearest neighbours",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="small",
        help="Name of nbody data partition [default, small]",
    )

    # Model parameters
    parser.add_argument(
        "--units", type=int, default=128, help="Number of values in the hidden layers"
    )
    parser.add_argument(
        "--lmax-hidden",
        type=int,
        default=1,
        help="Max degree of hidden representations.",
    )
    parser.add_argument(
        "--lmax-attributes",
        type=int,
        default=1,
        help="max degree of geometric attribute embedding",
    )
    parser.add_argument(
        "--layers", type=int, default=7, help="Number of message passing layers"
    )
    parser.add_argument(
        "--blocks", type=int, default=2, help="Number of layers in steerable MLPs."
    )
    # TODO instance norm
    parser.add_argument(
        "--norm",
        type=str,
        default="batch",
        help="Normalisation type [instance, batch]",
    )

    # wandb parameters
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Activate weights and biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="segnn",
        help="Weights and biases project",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="Weights and biases entity",
    )

    args = parser.parse_args()

    args.node_irreps = Irreps("2x1o + 1x0e")
    args.edge_irreps = Irreps("2x0e")

    # connect to wandb
    if args.wandb:
        wandb_name = "_".join(
            [
                args.wandb_project,
                args.dataset,
                args.dataset_name,
                args.target,
                str(int(time.time())),
            ]
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=args,
            entity=args.wandb_entity,
        )

    # Create hidden irreps
    hidden_irreps = weight_balanced_irreps(
        scalar_units=args.units,
        # attribute irreps
        irreps_right=Irreps.spherical_harmonics(args.lmax_attributes),
        use_sh=True,
        lmax=args.lmax_hidden,
    )

    # build model
    segnn = SEGNN(
        hidden_irreps=hidden_irreps,
        output_irreps=Irreps("1x1o"),
        num_layers=args.layers,
        task="node",
        blocks_per_layer=args.blocks,
        norm=args.norm,
    )
    segnn = hk.without_apply_rng(hk.transform_with_state(segnn))

    if args.dataset == "charged":
        dataset_train = ChargedDataset(
            partition="train",
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
            n_bodies=args.n_bodies,
        )
        dataset_val = ChargedDataset(
            partition="val", dataset_name=args.dataset_name, n_bodies=args.n_bodies
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

    train(segnn, dataset_train, dataset_val, dataset_test, args)
