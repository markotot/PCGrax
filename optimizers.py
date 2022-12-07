import functools
from typing import Callable, Optional, Union, NamedTuple, Any, Tuple, Dict, List

import chex
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
from optax._src.wrappers import ShouldSkipUpdateFunction
import functools
from typing import Callable, Optional, Union, NamedTuple, Any, Tuple, Dict, List

import chex
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from optax._src.wrappers import ShouldSkipUpdateFunction


def build_multi_step_optimizer(learning_rate):
    optimizer = optax.adam(learning_rate=learning_rate)
    multi_step_optimizer = optax.MultiSteps(optimizer, every_k_schedule=2)
    return multi_step_optimizer


class PCGraxState(NamedTuple):
    mini_step: jnp.ndarray
    gradient_step: jnp.ndarray
    inner_opt_state: Any
    acc_grads: Any
    grads_per_task: Any
    skip_state: chex.ArrayTree = ()


class PCGrax(optax.MultiSteps):
    def __init__(self, opt: optax.GradientTransformation,
                 every_k_schedule: Union[int, Callable[[jnp.ndarray], jnp.ndarray]],
                 use_grad_mean: bool = True,
                 should_skip_update_fn: Optional[ShouldSkipUpdateFunction] = None):

        super().__init__(opt, every_k_schedule, use_grad_mean, should_skip_update_fn)
        self.grads_per_task = {}

    def init(self, params, task_names: Any) -> PCGraxState:
        ms_state = super().init(params)

        for task_name in task_names:
            self.grads_per_task[task_name] = ms_state.acc_grads

        pcg_state = PCGraxState(mini_step=ms_state.mini_step,
                                gradient_step=ms_state.gradient_step,
                                inner_opt_state=ms_state.inner_opt_state,
                                acc_grads=ms_state.acc_grads,
                                grads_per_task=self.grads_per_task
                                )

        return pcg_state

    def update(self,
               updates: optax.Updates,
               state: PCGraxState,
               task: str,
               rng,
               params: Optional[optax.Params] = None) -> Tuple[optax.Updates, PCGraxState]:

        k_steps = self._every_k_schedule(state.gradient_step)
        acc_grads = jax.tree_util.tree_map(
            functools.partial(self._acc_update, n_acc=state.mini_step),
            updates, state.acc_grads)

        # Update the grads for the given task by calculating the running average
        grads_per_task = state.grads_per_task
        new_grad_tasks = jax.tree_util.tree_map(
            functools.partial(self._acc_update, n_acc=state.mini_step),
            updates, state.grads_per_task[task])
        grads_per_task[task] = new_grad_tasks

        should_skip_update, skip_state = self._should_skip_update_fn(
            updates, state.gradient_step, params)

        def final_step(args):
            del args

            # Flatten grads into vector
            raveled_grads_per_task = {k: ravel_pytree(grads) for k, grads in state.grads_per_task.items()}

            
            def project_grads(flattened_grads: Dict[str, jnp.ndarray], rng: chex.PRNGKey) -> List[jnp.ndarray]:
                """
                :param flattened_grads: Dictionary of 1D vectors representing gradients for each task
                :param rng: random key for shuffling the order of projections
                :return: returns the list of projected gradients
                """

                def apply_projection(i, info):
                    """
                    Project source vector to be orthogonal to v2 if they are conflicting, noop otherwise
                    :param v1: source vector
                    :param v2: target vector
                    :return: projected_vector
                    """
                    v1 = info['source']
                    grads = info['flattened_grads']
                    shuffled_order = info['shuffled_order']

                    idx = shuffled_order[i]
                    inner_product = jnp.dot(v1, grads[idx])
                    projection_direction = inner_product / jnp.dot(grads[idx], grads[idx])
                    v1 = v1 - jnp.minimum(projection_direction, 0) * grads[idx]

                    info['source'] = v1
                    return info

                shuffled_order = jax.random.permutation(rng, len(flattened_grads))  # Randomize the order of projections

                grad_length = len(flattened_grads[list(flattened_grads.keys())[0]][0])
                flattened_grads_new = jnp.zeros(shape=(len(flattened_grads.keys()), grad_length))
                for i, key in enumerate(flattened_grads.keys()):
                    flattened_grads_new = flattened_grads_new.at[i].set(flattened_grads[key][0])

                def apply_task_projections(source_task_grad):
                    info = {'source': source_task_grad,
                            'flattened_grads': flattened_grads_new,
                            'shuffled_order': shuffled_order}
                    info = jax.lax.fori_loop(lower=0, upper=len(shuffled_order), body_fun=apply_projection,
                                                         init_val=info)
                    return info['source']

                projected_grads = jax.vmap(apply_task_projections)(flattened_grads_new)
                return projected_grads

            projected_vectors = project_grads(raveled_grads_per_task, rng)
            projected_vector = jnp.mean(jnp.asarray(projected_vectors), axis=0)

            # raveled_grads_per_taks shape is (Dict[task_name, Tuple(grads, unravel_fn))
            unravel_fn = list(raveled_grads_per_task.values())[0][1]
            projected_grads = unravel_fn(projected_vector)

            # Do the update based on the projected gradients
            final_updates, new_inner_state = self._opt.update(
                projected_grads, state.inner_opt_state, params=params)

            new_state = PCGraxState(
                mini_step=jnp.zeros([], dtype=jnp.int32),
                gradient_step=optax.safe_int32_increment(state.gradient_step),
                inner_opt_state=new_inner_state,
                acc_grads=jax.tree_util.tree_map(jnp.zeros_like, acc_grads),
                grads_per_task=jax.tree_util.tree_map(jnp.zeros_like, grads_per_task),
                skip_state=skip_state)

            return final_updates, new_state

        def mid_step(args):
            del args
            updates_shape_dtype, _ = jax.eval_shape(
                self._opt.update, acc_grads, state.inner_opt_state, params=params)
            mid_updates = jax.tree_util.tree_map(
                lambda sd: jnp.zeros(sd.shape, sd.dtype), updates_shape_dtype)

            new_state = PCGraxState(
                mini_step=optax.safe_int32_increment(state.mini_step),
                gradient_step=state.gradient_step,
                inner_opt_state=state.inner_opt_state,
                acc_grads=acc_grads,
                grads_per_task=grads_per_task,
                skip_state=skip_state)

            return mid_updates, new_state

        new_updates, new_state = jax.lax.cond(
            state.mini_step < k_steps - 1, (), mid_step, (), final_step)

        if (should_skip_update.dtype, should_skip_update.shape) != (jnp.bool_, ()):
            raise ValueError(
                'The `should_skip_update_fn` function should return a boolean scalar '
                f'array, but it returned an array of dtype {should_skip_update.dtype}'
                f' and shape {should_skip_update.shape}')

        pcg_state_when_skip = PCGraxState(
            mini_step=state.mini_step,
            gradient_step=state.gradient_step,
            inner_opt_state=state.inner_opt_state,
            acc_grads=state.acc_grads,
            grads_per_task=state.grads_per_task,
            skip_state=skip_state)

        zero_updates = jax.tree_util.tree_map(jnp.zeros_like, updates)
        new_updates, new_state = jax.lax.cond(
            should_skip_update,
            (), lambda args: (zero_updates, pcg_state_when_skip),
            (), lambda args: (new_updates, new_state))

        return new_updates, new_state

    def has_updated(self, state: PCGraxState) -> jnp.ndarray:
        return jnp.logical_and(state.mini_step == 0, state.gradient_step > 0)
