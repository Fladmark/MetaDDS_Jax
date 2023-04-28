import jax
import jax.numpy as jnp

# Define the function
def rastrigin(v):
    return v**2 - 10*jnp.cos(2*jnp.pi*v) + 10

# Compute the gradient of the function
grad_rastrigin = jax.grad(rastrigin)

# Gradient descent algorithm
def gradient_descent(func, grad_func, v_init, lr=0.01, num_iters=10000):
    v = v_init
    for _ in range(num_iters):
        gradient = grad_func(v)
        v = v - lr * gradient
    return v

# Apply gradient descent
v_init = jnp.array(2.0)  # Initial value for v
optimized_v = gradient_descent(rastrigin, grad_rastrigin, v_init)
print(optimized_v)