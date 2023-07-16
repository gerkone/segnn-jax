import time
from functools import partial
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
from jax import jit

from segnn_jax import SteerableGraphsTuple


@partial(jit, static_argnames=["model_fn", "criterion", "task", "do_mask", "eval_trn"])
def loss_fn(
    params: hk.Params,
    state: hk.State,
    st_graph: SteerableGraphsTuple,
    target: jnp.ndarray,
    model_fn: Callable,
    criterion: Callable,
    task: str = "node",
    do_mask: bool = True,
    eval_trn: Callable = None,
) -> Tuple[float, hk.State]:
    pred, state = model_fn(params, state, st_graph)
    if eval_trn is not None:
        pred = eval_trn(pred)
    if task == "node":
        mask = jraph.get_node_padding_mask(st_graph.graph)
    if task == "graph":
        mask = jraph.get_graph_padding_mask(st_graph.graph)
    # broadcase mask for vector targets
    if len(pred.shape) == 2:
        mask = mask[:, jnp.newaxis]
    if do_mask:
        target = target * mask
        pred = pred * mask
    assert target.shape == pred.shape
    return jnp.sum(criterion(pred, target)) / jnp.count_nonzero(mask), state


@partial(jit, static_argnames=["loss_fn", "opt_update"])
def update(
    params: hk.Params,
    state: hk.State,
    graph: SteerableGraphsTuple,
    target: jnp.ndarray,
    opt_state: optax.OptState,
    loss_fn: Callable,
    opt_update: Callable,
) -> Tuple[float, hk.Params, hk.State, optax.OptState]:
    (loss, state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, state, graph, target
    )
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), state, opt_state


def evaluate(
    loader,
    params: hk.Params,
    state: hk.State,
    loss_fn: Callable,
    graph_transform: Callable,
) -> Tuple[float, float]:
    eval_loss = 0.0
    eval_times = 0.0
    for data in loader:
        graph, target = graph_transform(data, training=False)
        eval_start = time.perf_counter_ns()
        loss, _ = jax.lax.stop_gradient(loss_fn(params, state, graph, target))
        eval_loss += jax.block_until_ready(loss)
        eval_times += (time.perf_counter_ns() - eval_start) / 1e6
    return eval_times / len(loader), eval_loss / len(loader)


def train(
    key,
    segnn,
    loader_train,
    loader_val,
    loader_test,
    loss_fn,
    eval_loss_fn,
    graph_transform,
    args,
):
    init_graph, _ = graph_transform(next(iter(loader_train)))
    params, segnn_state = segnn.init(key, init_graph)

    print(
        f"Starting {args.epochs} epochs "
        f"with {hk.data_structures.tree_size(params)} parameters.\n"
        "Jitting..."
    )

    total_steps = args.epochs * len(loader_train)

    # set up learning rate and optimizer
    learning_rate = args.lr
    if args.lr_scheduling:
        learning_rate = optax.piecewise_constant_schedule(
            learning_rate,
            boundaries_and_scales={
                int(total_steps * 0.7): 0.1,
                int(total_steps * 0.9): 0.1,
            },
        )
    opt_init, opt_update = optax.adamw(
        learning_rate=learning_rate, weight_decay=args.weight_decay
    )

    model_fn = segnn.apply

    loss_fn = partial(loss_fn, model_fn=model_fn)
    eval_loss_fn = partial(eval_loss_fn, model_fn=model_fn)
    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)
    eval_fn = partial(evaluate, loss_fn=eval_loss_fn, graph_transform=graph_transform)

    opt_state = opt_init(params)
    avg_time = []
    best_val = 1e10

    for e in range(args.epochs):
        train_loss = 0.0
        train_start = time.perf_counter_ns()
        for data in loader_train:
            graph, target = graph_transform(data)
            loss, params, segnn_state, opt_state = update_fn(
                params=params,
                state=segnn_state,
                graph=graph,
                target=target,
                opt_state=opt_state,
            )
            train_loss += loss
        train_time = (time.perf_counter_ns() - train_start) / 1e6
        train_loss /= len(loader_train)
        print(
            f"[Epoch {e+1:>4}] train loss {train_loss:.6f}, epoch {train_time:.2f}ms",
            end="",
        )
        if e % args.val_freq == 0:
            eval_time, val_loss = eval_fn(loader_val, params, segnn_state)
            avg_time.append(eval_time)
            tag = ""
            if val_loss < best_val:
                best_val = val_loss
                tag = " (best)"
                _, test_loss_ckp = eval_fn(loader_test, params, segnn_state)
            print(f" - val loss {val_loss:.6f}{tag}, infer {eval_time:.2f}ms", end="")

        print()

    test_loss = 0
    _, test_loss = eval_fn(loader_test, params, segnn_state)
    # ignore compilation time
    avg_time = avg_time[2:]
    avg_time = sum(avg_time) / len(avg_time)
    print(
        "Training done.\n"
        f"Final test loss {test_loss:.6f} - checkpoint test loss {test_loss_ckp:.6f}.\n"
        f"Average (model) eval time {avg_time:.2f}ms"
    )
