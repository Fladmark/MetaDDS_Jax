import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, grad, value_and_grad
from sklearn.datasets import make_moons, make_circles, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import haiku as hk
import optax

# # Create the make_moons dataset
#X, y = make_circles(n_samples=2000, noise=0.1, factor=0.5)
# #X, y = make_moons(n_samples=1000, noise=0.1)
X, y = load_breast_cancer(return_X_y=True)

print(X[:, :15].shape)

print(X.shape)
print(y.shape)

X = jnp.array(X[:, :15], dtype=jnp.float32)
y = jnp.array(jnp.expand_dims(y, 1), dtype=jnp.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=422)



# Define the model
def model_fn(x):
    net = hk.Sequential([
        hk.Linear(4), jax.nn.relu,
        hk.Linear(1), jax.nn.sigmoid
    ])
    return net(x)

# Initialize the model
key = random.PRNGKey(42)
sample_input = jnp.array(X_train[0])
model = hk.transform(model_fn)
params = model.init(key, sample_input)



def loss_fn(params, x, y_true):
    y_pred = model.apply(params, None, x)
    return jnp.mean((y_true - y_pred) ** 2)


b = 1
w = None
for i in range(5000):
    key = random.PRNGKey(i)
    model = hk.transform(model_fn)
    params = model.init(key, sample_input)
    l = loss_fn(params, X_train, y_train)
    if l < b:
        w = params
        b = l

params = w



# Create the optimizer
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)

# Define the update function
@jit
def update(params, x, y_true, opt_state):
    grads = jax.grad(loss_fn)(params, x, y_true)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Train the model
epochs = 10000
for epoch in range(epochs):
    params, opt_state = update(params, X_train, y_train, opt_state)
    if epoch % 100 == 0:
        train_loss = loss_fn(params, X_train, y_train)
        print(f"Epoch: {epoch}, Loss: {train_loss}")

# Evaluate the model
def accuracy(params, x, y_true):
    y_pred = model.apply(params, None, x) > 0.5
    return jnp.mean(y_pred == y_true)

train_acc = accuracy(params, jnp.array(X_train), jnp.array(y_train))
test_acc = accuracy(params, jnp.array(X_test), jnp.array(y_test))
print(f"Train accuracy: {train_acc}, Test accuracy: {test_acc}")