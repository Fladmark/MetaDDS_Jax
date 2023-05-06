import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
import numpy as np

# Define the Rosenbrock function
def rosenbrock_function(x):
    return jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Gradient function using JAX
rosenbrock_gradient = jax.grad(rosenbrock_function)

# Initial point for optimization (random n-dimensional point)
n = 5  # Dimension
x0 = jnp.array([np.random.uniform() for _ in range(n)])

# Gradient descent settings
learning_rate = 0.001
num_iterations = 10000

# Gradient descent loop
x = x0
for i in range(num_iterations):
    grad = rosenbrock_gradient(x)
    x = x - learning_rate * grad

    # Print the progress
    if (i + 1) % 1000 == 0:
        print(f"Iteration {i + 1}: Loss={rosenbrock_function(x)}")

print("Optimized x:", x)
print("Final loss:", rosenbrock_function(x))