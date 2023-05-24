import jax
import jax.numpy as jnp
from jax import random

class SGLDOptimizer:
    def __init__(self, step_size, noise_scale, key=random.PRNGKey(0)):
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.key = key

    def step(self, grads, params):
        noise = random.normal(self.key, params.shape) * self.noise_scale
        self.key, _ = random.split(self.key)
        return params - self.step_size * grads + noise

    def update(self, params, grads):
        return self.step(params, grads)



# Define the function and its gradient
def f(x):
    return x**2

grad_f = jax.grad(f)

# Initialize the optimizer
sgld = SGLDOptimizer(step_size=0.01, noise_scale=0.1)

# Initial value
x_init = jnp.array(10.0)

# Training loop
for i in range(1000):
    grads = grad_f(x_init)
    x_init = sgld.update(grads, x_init)
    if i % 100 == 0:
        print(f"Iteration {i}, x: {x_init}, f(x): {f(x_init)}")