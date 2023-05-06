import jax
import jax.numpy as jnp
import numpy as np

def function(x):
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

def sgd_step(params, lr):
    grad = jax.grad(function)(params)
    params = params - lr * grad
    return params

def optimize_sgd(initial_params, learning_rate, num_steps):
    params = initial_params
    for step in range(num_steps):
        params = sgd_step(params, learning_rate)
        if step % 100 == 0:
            print(params)
            print(function(params))
    return params

# Example usage:
initial_params = jnp.array([1.0000148,  0.9964975,  0.99111277, 0.97567415])
#initial_params = jnp.array([0.9993484, 0.9986947, 0.9973851, 0.9947641])
#print(function(initial_params))
initial_params = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

learning_rate = 0.001
num_steps = 10000

optimized_params = optimize_sgd(initial_params, learning_rate, num_steps)
print("Optimized parameters:", optimized_params)

print(function(optimized_params))