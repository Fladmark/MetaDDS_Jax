import chex
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import haiku as hk
import optax
from optax._src import base
import numpy as np

# Define the simple neural network
def xor_model(x):
    return hk.Sequential([
        hk.Linear(2), jax.nn.sigmoid,
        hk.Linear(1),
    ])(x)

def loss_fn(params, x, y_true, model):
    y_pred = model.apply(params, None, x)
    return jnp.mean((y_true - y_pred) ** 2)

class xor_task:

    def __init__(self):
        self.model = hk.transform(xor_model)

        # Create the XOR dataset
        X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
        y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)


        # Split the dataset into train, validation, and test sets (use the same data in this example)
        self.X_train, X_val, X_test = X, X, X
        self.y_train, y_val, y_test = y, y, y

        # Initialize the model
        rng = jax.random.PRNGKey(42)
        self.params = self.model.init(rng, self.X_train)

    def get_loss(self, parameters):
        a, b, c, d, e, f, g, h, i = parameters
        new_params = {
            'linear': {
                'w': jnp.array([[a, b], [c, d]]),
                'b': jnp.array([g, h]),
            },
            'linear_1': {
                'w': jnp.array([[e], [f]]),
                'b': jnp.array([i]),
            },
        }
        for param in self.params:
            self.params[param] = new_params[param]

        return loss_fn(self.params, self.X_train, self.y_train, self.model)



# task = xor_task()
# print(task.params)
#
# b = 1
# for i in range(10000):
#     k = task.get_loss(np.random.rand(6)*5)
#     if k < b:
#         print(k)
#         b = k
# print(b)










# # Define the loss function
#
# # Set hyperparameters and optimizer
# learning_rate = 0.1
# epochs = 10000
# optimizer = optax.adam(learning_rate)
# opt_state = optimizer.init(params)
#
# # Training loop
# for epoch in range(epochs):
#     params, opt_state = update(params, X_train, y_train, opt_state)
#
#     if epoch % 1000 == 0:
#         train_loss = loss_fn(params, X_train, y_train, model)
#         val_loss = loss_fn(params, X_val, y_val)
#         print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
#
# # Test the model
# y_pred = model.apply(params, None, X_test)
# test_loss = loss_fn(params, X_test, y_test, model)
# print(f"Test Loss: {test_loss:.4f}")
# print(f"Predictions: {y_pred.round()}")
#
# print(params)