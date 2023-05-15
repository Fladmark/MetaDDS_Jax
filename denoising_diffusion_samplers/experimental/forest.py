import jax
import jax.numpy as jnp
import haiku as hk
import optax
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess Forest CoverTypes Dataset
covtype = fetch_covtype()
data = covtype["data"]
target = covtype["target"]

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Adjust the targets to be zero-based, as required by CrossEntropy loss
target -= 1

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Convert to jax arrays
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

# Define the model
def net_fn(x):
    net = hk.Sequential([
        hk.Linear(20), jax.nn.relu,
        hk.Linear(7),  # There are 7 classes in the dataset
    ])
    return net(x)

# Transform the model function into a pair of pure functions
net = hk.without_apply_rng(hk.transform(net_fn))

# Create optimizer
optimizer = optax.adam(0.01)

# Training loop
@jax.jit
def train_step(params, opt_state, x, y):
    # Compute gradients
    def loss_fn(params):
        logits = net.apply(params, x)
        return optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, 7)).mean()

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)

    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# Initialize parameters and optimizer
params = net.init(jax.random.PRNGKey(42), X_train[0])
opt_state = optimizer.init(params)

for parm in params:
    print(parm)

print(params["linear"]["w"].shape)
print(params["linear"]["b"].shape)
print(params["linear_1"]["w"].shape)
print(params["linear_1"]["b"].shape)


# # Training loop
# for epoch in range(1000):
#     params, opt_state = train_step(params, opt_state, X_train, y_train)
#     logits = net.apply(params, X_test)
#     pred = jnp.argmax(logits, axis=-1)
#     accuracy = (pred == y_test).mean()
#     print(f"Test accuracy: {accuracy}")
#
# # Evaluate on test data
# logits = net.apply(params, X_test)
# pred = jnp.argmax(logits, axis=-1)
# accuracy = (pred == y_test).mean()
# print(f"Test accuracy: {accuracy}")

