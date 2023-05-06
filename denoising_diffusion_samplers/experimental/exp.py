import jax
import jax.numpy as jnp
from jax import random
from sklearn.datasets import make_moons
import haiku as hk
import optax
import numpy as np

# Set up the make_moons task
def generate_moons_data(num_samples=50000):
    X, y = make_moons(n_samples=num_samples, noise=0.1)
    return X, y.reshape(-1, 1)

# Define the model
def task_model(x):
    HIDDENSIZE = 10
    return hk.Sequential([
        hk.Linear(HIDDENSIZE),
        jax.nn.relu,
        hk.Linear(1),
        jax.nn.sigmoid
    ])(x)

# Transforms
task_model = hk.without_apply_rng(hk.transform(task_model))

# Initialize parameters
DATASIZE = 2
GTSIZE = 1
key = random.PRNGKey(42)
x_sample = jnp.zeros((1, DATASIZE))
params = task_model.init(key, x_sample)

weights = params['linear']['w']
biases = params['linear']['b']
weights1 = params['linear_1']['w']
biases1 = params['linear_1']['b']

# Modify the weights and biases.
new_weights = np.random.randn(*weights.shape)
new_biases = np.random.randn(*biases.shape)
new_weights1 = np.random.randn(*weights1.shape)
new_biases1 = np.random.randn(*biases1.shape)

print(new_weights)
print(new_biases)
print(new_weights1)
print(new_biases1)



# Update the model parameters.
params['linear']['w'] = jnp.array(new_weights)
params['linear']['b'] = jnp.array(new_biases)
params['linear_1']['w'] = jnp.array(new_weights1)
params['linear_1']['b'] = jnp.array(new_biases1)



# Binary cross entropy loss
def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

# Set up the optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# Update function
@jax.jit
def update(params, opt_state, x, y):
    y_pred = task_model.apply(params, x)
    loss = jnp.mean(binary_cross_entropy_loss(y, y_pred))
    grads = jax.grad(lambda p, x, y: jnp.mean(binary_cross_entropy_loss(y, task_model.apply(p, x))))(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
X, y = generate_moons_data()
num_epochs = 10
batch_size = 64
num_batches = len(X) // batch_size

for epoch in range(num_epochs):
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        x_batch, y_batch = X[start:end], y[start:end]
        params, opt_state, loss = update(params, opt_state, x_batch, y_batch)
    print(f'Epoch {epoch + 1}/{num_epochs} Loss: {loss:.6f}')