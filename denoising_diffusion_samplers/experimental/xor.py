import chex
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import haiku as hk
import optax
from optax._src import base

# Define the simple neural network

def xor_model(x):
    return hk.Sequential([
        hk.Linear(2), jax.nn.sigmoid,
        hk.Linear(1),
    ])(x)

# Transform the model to work with Haiku
model = hk.transform(xor_model)

# Create the XOR dataset
X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)


# Split the dataset into train, validation, and test sets (use the same data in this example)
X_train, X_val, X_test = X, X, X
y_train, y_val, y_test = y, y, y

# Initialize the model
rng = jax.random.PRNGKey(42)
params = model.init(rng, X_train)


# Define the loss function
def loss_fn(params, x, y_true):
    y_pred = model.apply(params, None, x)
    return jnp.mean((y_true - y_pred) ** 2)




# b = 1
# for i in range(5000):
#     key = jax.random.PRNGKey(i)
#     model = hk.transform(xor_model)
#     params = model.init(key, X_train)
#     l = loss_fn(params, X_train, y_train)
#     if l < b:
#         b = l
# print(b)
#
# model = hk.transform(xor_model)
# params = model.init(rng, X_train)
# arr = []
# q = 1
# for i in range(10000):
#     key = jax.random.PRNGKey(i+10000)
#     a,b,c,d,e,f = (jax.random.normal(key, (1, 6)) * 1.075)[0]
#     new_params = {
#         'linear': {
#             'w': jnp.array([[a, b], [c, d]]),
#             'b': jnp.array([0, 0]),
#         },
#         'linear_1': {
#             'w': jnp.array([[e], [f]]),
#             'b': jnp.array([0]),
#         },
#     }
#     for param in params:
#         params[param] = new_params[param]
#
#     l = loss_fn(params, X_train, y_train)
#     arr.append(l)
#     if l < q:
#         q = l
# print(min(arr))
#
# exit()




# Define the update step
@jax.jit
def update(params, x, y_true, opt_state):
    grads = jax.grad(loss_fn)(params, x, y_true)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Set hyperparameters and optimizer
learning_rate = 0.1
epochs = 10000
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Training loop
for epoch in range(epochs):
    params, opt_state = update(params, X_train, y_train, opt_state)

    if epoch % 1000 == 0:
        train_loss = loss_fn(params, X_train, y_train)
        val_loss = loss_fn(params, X_val, y_val)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Test the model
y_pred = model.apply(params, None, X_test)
test_loss = loss_fn(params, X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Predictions: {y_pred.round()}")

print(params)