import functools
from typing import Callable, Optional, Union, NamedTuple, Any, Tuple, List, Dict

import chex
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from optax._src.wrappers import ShouldSkipUpdateFunction


def init_optimizer_fn(learning_rate, task_names):
    def build_pcgrax_optimizer(learning_rate):
        optimizer = optax.adam(learning_rate=learning_rate)
        pcgrax_multi_step_optimizer = PCGrax(optimizer, every_k_schedule=8, task_names=task_names)
        return pcgrax_multi_step_optimizer

    optimizer = optax.inject_hyperparams(build_pcgrax_optimizer)(learning_rate)
    return optimizer


class PCGraxState(NamedTuple):
    mini_step: jnp.ndarray
    gradient_step: jnp.ndarray
    inner_opt_state: Any
    grads_per_task: Any
    task_idx: jnp.int_
    projection_rng: jax.random.PRNGKey
    skip_state: chex.ArrayTree = ()


class PCGrax(optax.MultiSteps):
    def __init__(self, opt: optax.GradientTransformation,
                 every_k_schedule: Union[int, Callable[[jnp.ndarray], jnp.ndarray]],
                 task_names: List[str],
                 use_grad_mean: bool = True,
                 should_skip_update_fn: Optional[ShouldSkipUpdateFunction] = None):

        super().__init__(opt, every_k_schedule, use_grad_mean, should_skip_update_fn)
        self.grads_per_task = None  # task_name -> gradients
        self.num_tasks = len(task_names)
        self.task_grad_length = None

    def init(self, params) -> PCGraxState:
        ms_state = super().init(params)

        unraveled_grads, _ = ravel_pytree(ms_state.acc_grads)
        self.grads_per_task = jnp.zeros(shape=(self.num_tasks, len(unraveled_grads)))
        self.task_grad_length = len(unraveled_grads)
        pcg_state = PCGraxState(mini_step=ms_state.mini_step,
                                gradient_step=ms_state.gradient_step,
                                inner_opt_state=ms_state.inner_opt_state,
                                grads_per_task=self.grads_per_task,
                                task_idx=None,
                                projection_rng=None,
                                )
        return pcg_state

    def update(self,
               updates: optax.Updates,
               state: PCGraxState,
               params: Optional[optax.Params] = None) -> Tuple[optax.Updates, PCGraxState]:

        rng = state.projection_rng
        task_idx = state.task_idx

        raveled_updates, unravel_fn = ravel_pytree(updates)
        k_steps = self._every_k_schedule(state.gradient_step)


        # Update the grads for the given task by calculating the running average
        grads_per_task = state.grads_per_task
        running_average_fn = functools.partial(self._acc_update, n_acc=state.mini_step)
        new_grads_for_task = running_average_fn(raveled_updates, state.grads_per_task[task_idx])
        grads_per_task = grads_per_task.at[task_idx].set(new_grads_for_task)

        should_skip_update, skip_state = self._should_skip_update_fn(
            updates, state.gradient_step, params)

        def final_step(args):
            del args
            def project_grads(raveled_grads: jnp.ndarray, rng: chex.PRNGKey) -> List[jnp.ndarray]:
                """
                :param flattened_grads: Dictionary of 1D vectors representing gradients for each task
                :param rng: random key for shuffling the order of projections
                :return: returns the list of projected gradients
                """

                def apply_single_projection(n, info):
                    idx = info['projection_order'][n]
                    v1 = info['source_grads']  # Gets the grad that will be projected
                    v2 = info['all_grads'][idx]  # Gets the target grad to project to

                    inner_product = jnp.dot(v1, v2)
                    projection_direction = inner_product / jnp.dot(v2, v2)
                    v1 = v1 - jnp.minimum(projection_direction, 0) * v2

                    info['source_grads'] = v1
                    return info

                def apply_task_projections(source_task_grad, all_grads, rng):
                    projections_order = jax.random.permutation(rng, len(all_grads))
                    info = {'source_grads': source_task_grad,
                            'all_grads': all_grads,
                            'projection_order': projections_order,}
                    info = jax.lax.fori_loop(lower=0, upper=len(projections_order), body_fun=apply_single_projection,
                                             init_val=info)
                    return info['source_grads']

                rng, *rng_list = jax.random.split(rng, len(raveled_grads) + 1)
                projected_grads = jax.vmap(apply_task_projections, in_axes=(0, None, 0))(raveled_grads,
                                                                                         raveled_grads,
                                                                                         jnp.stack(rng_list))
                return projected_grads

            # Do gradient projection and average them out
            projected_grads = project_grads(grads_per_task, rng)
            projected_grads = jnp.mean(projected_grads, axis=0)
            projected_grads = unravel_fn(projected_grads)

            # Do the update based on the projected gradients
            final_updates, new_inner_state = self._opt.update(
                projected_grads, state.inner_opt_state, params=params)

            new_state = PCGraxState(
                mini_step=jnp.zeros([], dtype=jnp.int32),
                gradient_step=optax.safe_int32_increment(state.gradient_step),
                inner_opt_state=new_inner_state,
                grads_per_task=jnp.zeros_like(grads_per_task),
                task_idx=task_idx,
                projection_rng=None,
                skip_state=skip_state)

            return final_updates, new_state

        def mid_step(args):
            del args

            mid_updates = jnp.zeros(shape=(len(grads_per_task[0]),))
            mid_updates = unravel_fn(mid_updates)
            new_state = PCGraxState(
                mini_step=optax.safe_int32_increment(state.mini_step),
                gradient_step=state.gradient_step,
                inner_opt_state=state.inner_opt_state,
                grads_per_task=grads_per_task,
                task_idx=task_idx,
                projection_rng=None,
                skip_state=skip_state)

            return mid_updates, new_state

        new_updates, new_state = jax.lax.cond(
            state.mini_step < k_steps - 1, (), mid_step, (), final_step)

        # This shouldn't work while traced right?!?
        if (should_skip_update.dtype, should_skip_update.shape) != (jnp.bool_, ()):
            raise ValueError(
                'The `should_skip_update_fn` function should return a boolean scalar '
                f'array, but it returned an array of dtype {should_skip_update.dtype}'
                f' and shape {should_skip_update.shape}')

        pcg_state_when_skip = PCGraxState(
            mini_step=state.mini_step,
            gradient_step=state.gradient_step,
            inner_opt_state=state.inner_opt_state,
            grads_per_task=state.grads_per_task,
            task_idx=state.task_idx,
            projection_rng=None,
            skip_state=skip_state)

        zero_updates = jax.tree_util.tree_map(jnp.zeros_like, updates)
        new_updates, new_state = jax.lax.cond(
            should_skip_update,
            (), lambda args: (zero_updates, pcg_state_when_skip),
            (), lambda args: (new_updates, new_state))

        return new_updates, new_state
