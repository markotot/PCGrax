import functools
from functools import reduce
from typing import Callable, Optional, Union, NamedTuple, Any, Tuple, Dict, List

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax._src.wrappers import ShouldSkipUpdateFunction


def build_multi_step_optimizer(learning_rate):
    optimizer = optax.adam(learning_rate=learning_rate)
    multi_step_optimizer = optax.MultiSteps(optimizer, every_k_schedule=2)
    return multi_step_optimizer


class PCGradOptimizer:

    def __init__(self, params, task_names):
        self.acc_grads = {}
        for task_name in task_names:
            self.acc_grads[task_name] = params
            self.acc_grads[task_name] = jax.tree_util.tree_map(lambda x: 0, self.acc_grads[task_name])

    def update(self, grads, task):
        self.acc_grads[task] = jax.tree_util.tree_map(lambda x, y: x + y, self.acc_grads[task], grads)

    def accumulate(self):
        pass


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
            def flatten_grads(grads: Dict[str, chex.ArrayTree]) -> Tuple[
                Dict[str, jnp.ndarray], chex.PyTreeDef, List[int]]:
                """

                :param grads: Gradients for each task, each task represented by a PyTree
                :return:
                        flattened grads: Gradients for each task, each task represented by a 1D vector
                        tree_def: The information of the PyTree structure, to be used to unflatten later
                        layer_shapes: Shape of each layer in the Network, to be used to unflatten later
                """
                flattened_grads = {}
                # For each task in the gradient unflatten the tree to a list of layers
                for task_name in grads:
                    grad_vector = jnp.zeros(shape=(0,))
                    list_of_layers, _ = jax.tree_util.tree_flatten(grads[task_name])  # transform Grad to List of Layers
                    for layer in list_of_layers:
                        grad_vector = jnp.concatenate((grad_vector, layer.flatten()))  # transform Layer to 1D vector
                    flattened_grads[task_name] = grad_vector

                # Get tree_def and layer_shapes
                layered_grads, tree_def = jax.tree_util.tree_flatten(list(grads.values())[0])
                layer_shapes = []
                for layer in layered_grads:
                    layer_shapes.append(layer.shape)

                return flattened_grads, tree_def, layer_shapes

            flatten_grads, tree_def, layer_shapes = flatten_grads(grads=state.grads_per_task)

            def project_grads(flattened_grads: Dict[str, jnp.ndarray], rng: chex.PRNGKey) -> List[jnp.ndarray]:
                """

                :param flattened_grads: Dictionary of 1D vectors representing gradients for each task
                :param rng: random key for shuffling the order of projections
                :return: returns the list of projected gradients
                """

                def apply_projection(v1, v2):
                    """
                    Project source vector to be orthogonal to v2 if they are conflicting, noop otherwise
                    :param v1: source vector
                    :param v2: target vector
                    :return: projected_vector
                    """
                    inner_product = jnp.dot(v1, v2)
                    projection_direction = inner_product / jnp.dot(v2, v2)
                    v1 = v1 - jnp.minimum(projection_direction, 0) * v2
                    return v1

                shuffled_order = jax.random.permutation(rng, len(flattened_grads))  # Randomize the order of projections

                projected_grads = []
                # For each source gradient
                for task_source in flattened_grads:
                    source_task_grad = flattened_grads[task_source]
                    # Project the source to every gradient
                    # It's ok to project a vector to itself (v1->v1), nothing will change
                    for k in shuffled_order:
                        target_task_grad = list(flattened_grads.values())[k]
                        source_task_grad = apply_projection(source_task_grad, target_task_grad)
                    projected_grads.append(source_task_grad)
                return projected_grads

            projected_vectors = project_grads(flatten_grads, rng)
            projected_vector = jnp.mean(jnp.asarray(projected_vectors), axis=0)

            def restore_original_shape(projected_vector: jnp.ndarray, tree_def: chex.PyTreeDef,
                                       layer_shapes: List[int]):
                """
                Takes in a gradient in a form of the vector, and unflattens it to represent the original gradient shape
                :param projected_vector: Accumulated projected gradient vector
                :param tree_def: Original definition of the gradient's PyTree structure
                :param layer_shapes: Shapes of the network layers
                :return: Accumulated projected gradient in a PyTree form
                """
                # Determine respective layer's start/end indices in the 1D vector based on the layer shapes
                layer_sizes = []
                for layer_shape in layer_shapes:
                    layer_size = reduce(lambda x, y: x * y, np.asarray(layer_shape))
                    layer_sizes.append(layer_size)
                layer_sections = np.concatenate((np.zeros(1, dtype=int), np.cumsum(layer_sizes)))

                # Create the list of layers based on the start/end indices
                reshaped_layers = []
                for i in range(1, len(layer_sections)):
                    start = layer_sections[i - 1]
                    end = layer_sections[i]
                    layer_section = projected_vector[start:end]
                    reshaped_layer = layer_section.reshape(layer_shapes[i - 1])
                    reshaped_layers.append(reshaped_layer)

                # Unflatten into a PyTree given tree_def and the list of layers
                task_grad_pytree = jax.tree_unflatten(tree_def, reshaped_layers)
                return task_grad_pytree

            projected_grads = restore_original_shape(projected_vector, tree_def, layer_shapes)

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
