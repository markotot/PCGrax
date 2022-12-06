from tensorflow import keras
import jax.numpy as jnp

def load_data(dataset_name):

    if dataset_name == 'fashion':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    elif dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    else:
        assert False

    x_train, x_test, y_train, y_test = jnp.array(x_train, dtype=jnp.float32), \
        jnp.array(x_test, dtype=jnp.float32), \
        jnp.array(y_train, dtype=jnp.float32), \
        jnp.array(y_test, dtype=jnp.float32)

    x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)

    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, x_test, y_train, y_test
