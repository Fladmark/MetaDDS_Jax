import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, grad, value_and_grad
from sklearn.datasets import make_moons, make_circles, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import haiku as hk
import optax

def model_fn(x):
    net = hk.Sequential([
        hk.Linear(4), jax.nn.relu,
        hk.Linear(1), jax.nn.sigmoid
    ])
    return net(x)

def loss_fn(params, x, y_true, model):
    y_pred = model.apply(params, None, x)
    return jnp.mean((y_true - y_pred) ** 2)

def accuracy(params, x, y_true, model):
    y_pred = model.apply(params, None, x) > 0.5
    return jnp.mean(y_pred == y_true)

# Define the update function
@jit
def update(params, x, y_true, opt_state, optimizer):
    grads = jax.grad(loss_fn)(params, x, y_true)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

class breast_task:

    def __init__(self):
        X, y = load_breast_cancer(return_X_y=True)
        X = jnp.array(X[:, :15], dtype=jnp.float32)
        y = jnp.array(jnp.expand_dims(y, 1), dtype=jnp.float32)
        self.X_train, X_test, self.y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=422)

        # Define the model

        # Initialize the model
        self.key = random.PRNGKey(42)
        self.sample_input = jnp.array(self.X_train[0])
        self.model = hk.transform(model_fn)
        self.params = self.model.init(self.key, self.sample_input)

    def get_loss(self, parameters):
        l, l1 = parameters[:60].reshape(15,4), parameters[60:].reshape(4,1)
        self.params["linear"]["w"] = l
        self.params["linear_1"]["w"] = l1
        return loss_fn(self.params, self.X_train, self.y_train, self.model)

    def set_weight(self, parameters):
        l, l1 = parameters[:60].reshape(15,4), parameters[60:].reshape(4,1)
        self.params["linear"]["w"] = l
        self.params["linear_1"]["w"] = l1

    def reset_weight(self, key_number=None):
        if key_number:
            self.params = self.model.init(random.PRNGKey(key_number), self.sample_input)
        else:
            self.params = self.model.init(self.key, self.sample_input)

    def fine_tune(self):
        optimizer = optax.adam(0.001)
        opt_state = optimizer.init(self.params)

        # Train the model
        epochs = 10000
        for epoch in range(epochs):
            params, opt_state = update(params, self.X_train, self.y_train, opt_state, optimizer)
            if epoch % 100 == 0:
                train_loss = loss_fn(params, self.X_train, self.y_train)
                print(f"Epoch: {epoch}, Loss: {train_loss}")

        # Evaluate the model

        train_acc = accuracy(params, jnp.array(self.X_train), jnp.array(self.y_train))
        test_acc = accuracy(params, jnp.array(self.X_test), jnp.array(self.y_test))
        print(f"Train accuracy: {train_acc}, Test accuracy: {test_acc}")