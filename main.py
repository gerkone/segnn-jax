import argparse
import time
from functools import partial
from typing import Tuple, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import optax

import wandb
from experiments import setup_datasets
from segnn_jax import SEGNN, SteerableGraphsTuple, weight_balanced_irreps

key = jax.random.PRNGKey(0)


@jax.jit
def predict(
    params: hk.Params,
    state: hk.State,
    graph: SteerableGraphsTuple,
    mean_shift: Union[jnp.array, float] = 0,
    mad_shift: Union[jnp.array, float] = 1,
) -> Tuple[jnp.ndarray, hk.State]:
    pred, state = segnn.apply(params, state, graph)
    return (jnp.multiply(pred, mad_shift) + mean_shift), state


@partial(jax.jit, static_argnames=["mean_shift", "mad_shift", "mask_last"])
def mae(
    params: hk.Params,
    state: hk.State,
    graph: SteerableGraphsTuple,
    target: jnp.ndarray,
    mean_shift: Union[jnp.array, float] = 0,
    mad_shift: Union[jnp.array, float] = 1,
    mask_last: bool = False,
) -> Tuple[float, hk.State]:
    pred, state = predict(params, state, graph, mean_shift, mad_shift)
    assert target.shape == pred.shape
    # similar to get_graph_padding_mask
    if mask_last:
        return (jnp.abs(pred[:-1] - target[:-1])).mean(), state
    else:
        return (jnp.abs(pred - target)).mean(), state


@partial(jax.jit, static_argnames=["mean_shift", "mad_shift", "mask_last"])
def mse(
    params: hk.Params,
    state: hk.State,
    graph: SteerableGraphsTuple,
    target: jnp.ndarray,
    mean_shift: Union[jnp.array, float] = 0,
    mad_shift: Union[jnp.array, float] = 1,
    mask_last: bool = False,
) -> Tuple[float, hk.State]:
    pred, state = predict(params, state, graph, mean_shift, mad_shift)
    assert target.shape == pred.shape
    if mask_last:
        return (jnp.power(pred[:-1] - target[:-1], 2)).mean(), state
    else:
        return (jnp.power(pred - target, 2)).mean(), state


def train(
    segnn: hk.Transformed, loader_train, loader_val, loader_test, graph_transform, args
):
    print(f"Starting {args.epochs} epochs on {args.dataset}.")

    print("Jitting...")
    init_graph, _ = graph_transform(next(iter(loader_train)))
    params, segnn_state = segnn.init(key, init_graph)

    total_steps = args.epochs * len(loader_train)

    # set up learning rate and optimizer
    if args.lr_scheduling:
        learning_rate = optax.piecewise_constant_schedule(
            args.lr,
            boundaries_and_scales={
                int(total_steps * 0.8): 0.1,
                int(total_steps * 0.9): 0.1,
            },
        )
    else:
        learning_rate = args.lr

    opt_init, opt_update = optax.adamw(
        learning_rate=learning_rate, weight_decay=args.weight_decay
    )

    if args.dataset == "qm9":
        # for target normalization
        target_mean, target_mad = loader_train.dataset.calc_stats()
        # ignore padded target
        loss_fn = partial(mae, mask_last=True)
    else:
        # no normalization for nbody
        target_mean, target_mad = 0, 1
        loss_fn = mse

    @jax.jit
    def update(
        params: hk.Params,
        state: hk.State,
        graph: SteerableGraphsTuple,
        target: jnp.ndarray,
        opt_state: optax.OptState,
    ) -> Tuple[float, hk.Params, hk.State, optax.OptState]:
        (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, graph, target
        )
        updates, opt_state = opt_update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), state, opt_state

    opt_state = opt_init(params)
    avg_time = []

    for e in range(args.epochs):
        train_loss = 0.0
        train_start = time.perf_counter_ns()
        for data in loader_train:
            graph, target = graph_transform(data)
            target = jnp.divide(target - target_mean, target_mad).at[-1].set(0)
            loss, params, segnn_state, opt_state = update(
                params, segnn_state, graph, target, opt_state
            )
            train_loss += loss
        train_time = (time.perf_counter_ns() - train_start) / 1e6 / len(loader_train)
        train_loss /= len(loader_train)
        wandb_logs = {"train_loss": float(train_loss), "update_time": float(train_time)}
        print(
            "[Epoch {:>4}] training loss {:.6f}{}, update time {:.3f}ms".format(
                e + 1,
                train_loss,
                (" (normalized)" if args.dataset == "qm9" else ""),
                train_time,
            ),
            end="",
        )
        if e % args.val_freq == 0:
            val_loss = 0
            eval_start = time.perf_counter_ns()
            for data in loader_val:
                graph, target = graph_transform(data)
                loss, segnn_state = jax.lax.stop_gradient(
                    loss_fn(params, segnn_state, graph, target, target_mean, target_mad)
                )
                val_loss += loss
            eval_time = (time.perf_counter_ns() - eval_start) / 1e6 / len(loader_val)
            avg_time.append(eval_time)
            val_loss /= len(loader_val)
            wandb_logs.update(
                {"val_loss": float(val_loss), "eval_time": float(eval_time)}
            )
            print(
                " - validation loss {:.6f}, eval time {:.3f}ms".format(
                    val_loss, eval_time
                ),
                end="",
            )
        print()
        if args.wandb:
            wandb.log(wandb_logs)

    test_loss = 0
    for data in loader_test:
        graph, target = graph_transform(data)
        loss, segnn_state = jax.lax.stop_gradient(
            loss_fn(params, segnn_state, graph, target, target_mean, target_mad)
        )
        test_loss += loss
    test_loss /= len(loader_test)
    avg_time = sum(avg_time) / len(avg_time)
    if args.wandb:
        wandb.log({"test_loss": float(test_loss), "avg_eval_time": float(avg_time)})
    print(
        "Training done. Test loss {:.6f} - "
        "average eval time {:.3f}ms".format(test_loss, avg_time)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (number of graphs).",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--lr-scheduling",
        action="store_true",
        help="Use learning rate scheduling",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-12, help="Weight decay"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["qm9", "charged", "gravity"],
        help="Dataset name",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Maximum number of samples in nbody dataset",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=10,
        help="Evaluation frequency (number of epochs)",
    )

    # nbody parameters
    parser.add_argument(
        "--target",
        type=str,
        default="pos",
        help="Target. e.g. pos, force (gravity), alpha (qm9)",
    )
    parser.add_argument(
        "--neighbours",
        type=int,
        default=20,
        help="Number of connected nearest neighbours",
    )
    parser.add_argument(
        "--n-bodies",
        type=int,
        default=5,
        help="Number of bodies in the dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="small",
        choices=["small", "default", "small_out_dist"],
        help="Name of nbody data partition: default (200 steps), small (1000 steps)",
    )

    # qm9 parameters
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Radius (Angstrom) between which atoms to add links.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="one_hot",
        choices=["one_hot", "cormorant", "gilmer"],
        help="Type of input feature",
    )

    # Model parameters
    parser.add_argument(
        "--units", type=int, default=128, help="Number of values in the hidden layers"
    )
    parser.add_argument(
        "--lmax-hidden",
        type=int,
        default=2,
        help="Max degree of hidden representations.",
    )
    parser.add_argument(
        "--lmax-attributes",
        type=int,
        default=3,
        help="Max degree of geometric attribute embedding",
    )
    parser.add_argument(
        "--layers", type=int, default=7, help="Number of message passing layers"
    )
    parser.add_argument(
        "--blocks", type=int, default=2, help="Number of layers in steerable MLPs."
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="batch",
        choices=["instance", "batch", "none"],
        help="Normalisation type",
    )
    parser.add_argument(
        "--double-precision",
        action="store_true",
        help="Use double precision in model",
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

    # if specified set jax in double precision
    jax.config.update("jax_enable_x64", args.double_precision)

    # connect to wandb
    if args.wandb:
        wandb_name = "_".join(
            [
                args.wandb_project,
                args.dataset,
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

    # feature representations
    if args.dataset == "qm9":
        task = "graph"
        if args.feature_type == "one_hot":
            args.node_irreps = e3nn.Irreps("5x0e")
        elif args.feature_type == "cormorant":
            args.node_irreps = e3nn.Irreps("15x0e")
        elif args.feature_type == "gilmer":
            args.node_irreps = e3nn.Irreps("11x0e")
        args.output_irreps = e3nn.Irreps("1x0e")
        args.additional_message_irreps = e3nn.Irreps("1x0e")
    elif args.dataset in ["charged", "gravity"]:
        task = "node"
        args.node_irreps = e3nn.Irreps("2x1o + 1x0e")
        args.output_irreps = e3nn.Irreps("1x1o")
        args.additional_message_irreps = e3nn.Irreps("2x0e")

    # Create hidden irreps
    hidden_irreps = weight_balanced_irreps(
        scalar_units=args.units,
        # attribute irreps
        irreps_right=e3nn.Irreps.spherical_harmonics(args.lmax_attributes),
        use_sh=True,
        lmax=args.lmax_hidden,
    )

    # build model
    segnn = lambda x: SEGNN(
        hidden_irreps=hidden_irreps,
        output_irreps=args.output_irreps,
        num_layers=args.layers,
        task=task,
        pool="avg",
        blocks_per_layer=args.blocks,
        norm=args.norm,
    )(x)
    segnn = hk.without_apply_rng(hk.transform_with_state(segnn))

    dataset_train, dataset_val, dataset_test, graph_transform = setup_datasets(args)

    train(segnn, dataset_train, dataset_val, dataset_test, graph_transform, args)
