import haiku as hk
import jax
import jax.numpy as jnp
import optax
from optax._src.schedule import InjectHyperparamsState

from model import ConvNet
from optimizers import init_optimizer_fn, PCGraxState
from prepare_data import load_data

# jax.config.update("jax_disable_jit", True)
learning_rate = 0.001
task_names = ['mnist', 'fashion']
optimizer = init_optimizer_fn(learning_rate=learning_rate, task_names=task_names)


def cross_entropy_loss(weights, input_data, actual):
    preds = conv_net.apply(weights, rng, input_data)
    one_hot_actual = jax.nn.one_hot(actual, num_classes=10)
    log_preds = jnp.log(preds)
    return - jnp.mean(one_hot_actual * log_preds)


def batch_update_iteration(train_x, train_y, model_params, task_idx, optimizer_state, rng):
    loss, grads = jax.value_and_grad(cross_entropy_loss)(model_params, train_x, train_y)

    # Add task_idx and rng to the optimizer state
    opt_inner_state = PCGraxState(
        mini_step=optimizer_state.inner_state.mini_step,
        gradient_step=optimizer_state.inner_state.gradient_step,
        inner_opt_state=optimizer_state.inner_state.inner_opt_state,
        grads_per_task=optimizer_state.inner_state.grads_per_task,
        task_idx=task_idx,
        projection_rng=rng,
        skip_state=optimizer_state.inner_state)

    optimizer_state = InjectHyperparamsState(
        count=optimizer_state.count,
        hyperparams=optimizer_state.hyperparams,
        inner_state=opt_inner_state,
    )
    network_updates, new_optimizer_state = optimizer.update(updates=grads, state=optimizer_state)
    new_params = optax.apply_updates(model_params, network_updates)
    return loss, grads, new_params, new_optimizer_state


if __name__ == "__main__":

    rng = jax.random.PRNGKey(42)
    batch_size = 16
    epochs = 100
    # Load data
    mnist_train_x, mnist_test_x, mnist_train_y, mnist_test_y = load_data('mnist')
    fashion_train_x, fashion_test_x, fashion_train_y, fashion_test_y = load_data('fashion')

    # Create and initialize model
    model = conv_net = hk.transform(ConvNet)
    params = conv_net.init(rng, mnist_train_x[:1])

    # Testing PCGrad
    optimizer_state = optimizer.init(params)

    for i in range(1, epochs + 1):
        batches = jnp.arange((mnist_train_x.shape[0] // batch_size) + 1)  ### Batch Indices

        losses_mnist = []  ## Record loss of each batch
        losses_fashion = []  ## Record loss of each batch
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch * batch_size), int(batch * batch_size + batch_size)
            else:
                start, end = int(batch * batch_size), None

            jitted_batch_updated_iteration = jax.jit(batch_update_iteration)
            # Batch update iteration MNIST
            rng, rng_update = jax.random.split(rng, 2)
            loss_mnist, grads_mnist, params, optimizer_state = jitted_batch_updated_iteration(
                mnist_train_x[start:end],
                mnist_train_y[start:end],
                model_params=params,
                task_idx=0,
                optimizer_state=optimizer_state,
                rng=rng_update
            )

            # Batch update iteration FASHION
            rng, rng_update = jax.random.split(rng, 2)

            loss_fashion, grads_fashion, params, optimizer_state = jitted_batch_updated_iteration(
                fashion_train_x[start:end],
                fashion_train_y[start:end],
                model_params=params,
                task_idx=1,
                optimizer_state=optimizer_state,
                rng=rng_update
            )

            losses_mnist.append(loss_mnist.mean())
            losses_fashion.append(loss_fashion.mean())
            print("MNIST   - Loss : {:.3f}".format(jnp.array(losses_mnist)[-1]))
            print("FASHION - Loss : {:.3f}".format(jnp.array(losses_fashion)[-1]))
            print("")
