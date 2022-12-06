import jax.random
import jax.numpy as jnp

eps = 1e-5

def cosine_similarity(v1, v2):
    norm_v1 = jnp.linalg.norm(v1)
    norm_v2 = jnp.linalg.norm(v2)
    result = jnp.clip(jnp.dot(v1, v2) / (norm_v1 * norm_v2), -1, 1)
    return result


def apply_projection(source_task_grad, target_task_grad):
    inner_product = jnp.dot(source_task_grad, target_task_grad)
    projection_direction = inner_product / jnp.dot(target_task_grad, target_task_grad)
    source_task_grad = source_task_grad - jnp.minimum(projection_direction,
                                                      0) * target_task_grad
    return source_task_grad


def test_cosine_similarity_same_vector():
    rng = jax.random.PRNGKey(42)
    v1 = jax.random.normal(rng, shape=(10000,))
    assert -eps < cosine_similarity(v1, v1) - 1 < eps


def test_cosine_similarity_conflicting():
    v1 = jnp.array([1, 0])
    v2 = jnp.array([-1, 0])
    assert -eps < cosine_similarity(v1, v2) + 1 < eps


def test_cosine_similarit_normal():
    v1 = jnp.array([1, 1])
    v2 = jnp.array([1, -1])
    assert -eps < cosine_similarity(v1, v2) < eps


def test_projection_random():
    rng = jax.random.PRNGKey(1)
    rng, rng_v1, rng_v2 = jax.random.split(rng, 3)

    v1 = jax.random.normal(rng_v1, shape=(10000,))
    v2 = jax.random.normal(rng_v2, shape=(10000,))
    v1_pc = apply_projection(v1, v2)

    print(cosine_similarity(v1, v2))
    print(cosine_similarity(v1, v1_pc))
    assert -eps < cosine_similarity(v1, v1) - 1 < eps
    assert cosine_similarity(v1_pc, v2) >= -eps


def test_projection_conflicting():

    v1 = jnp.array([1, 1])
    v2 = jnp.array([0, -1])
    v1_pc = apply_projection(v1, v2)

    assert cosine_similarity(v1, v2) < -eps  # before projection
    assert cosine_similarity(v1_pc, v2) >= -eps  # after projection


test_cosine_similarity_same_vector()
test_cosine_similarity_conflicting()
test_cosine_similarit_normal()
test_projection_random()
test_projection_conflicting()
