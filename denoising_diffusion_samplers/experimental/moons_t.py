import chex
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import haiku as hk
import optax
from optax._src import base
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from sklearn.datasets import make_moons, make_circles, make_blobs
import haiku as hk
import optax
from sklearn.model_selection import train_test_split


# Define the simple neural network
def task_model(x):
    HIDDENSIZE = 10
    return hk.Sequential([
        hk.Linear(HIDDENSIZE),
        jax.nn.relu,
        hk.Linear(1),
        jax.nn.sigmoid
    ])(x)

def generate_moons_data(num_samples=10000):
    # normal 0.05 noise
    X, y = make_circles(n_samples=num_samples, factor=0.7, noise=0.05, random_state=0) #make_blobs(n_samples=num_samples, centers=2, random_state=42) #
    return X, y.reshape(-1, 1)

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

class moon_task:

    def __init__(self):
        # Transforms
        self.task_model = hk.without_apply_rng(hk.transform(task_model))
        self.X, self.y = generate_moons_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=0.5,random_state=42)

        self.idx = 0
        self.batch_size = 24

        # Initialize parameters
        DATASIZE = 2
        GTSIZE = 1
        key = random.PRNGKey(42)
        x_sample = jnp.zeros((1, DATASIZE))
        self.params = self.task_model.init(key, x_sample)

    def get_loss(self, parameters, type="training"):
        l,b, l1, b1 = parameters[:20].reshape(2,10), parameters[20:30].reshape(10), parameters[30:40].reshape(10,1), parameters[40:].reshape(1)
        self.params["linear"]["w"] = jnp.array(l)
        self.params["linear_1"]["w"] = jnp.array(l1)
        self.params["linear"]["b"] = jnp.array(b)
        self.params["linear_1"]["b"] = jnp.array(b1)

        if type == "validation":
            y_pred = self.task_model.apply(self.params, self.X_val)
            loss = jnp.mean(binary_cross_entropy_loss(self.y_val, y_pred))
        elif type == "test":
            y_pred = self.task_model.apply(self.params, self.X_test)
            loss = jnp.mean(binary_cross_entropy_loss(self.y_test, y_pred))
        elif type == "non-batch":
            y_pred = self.task_model.apply(self.params, self.X_train)
            loss = jnp.mean(binary_cross_entropy_loss(self.y_train, y_pred))
        else:
            indecies = np.array(range(self.idx, self.idx + self.batch_size)) % (self.X_train.shape[0] - 1)
            self.idx += self.batch_size
            if self.idx > self.X.shape[0] - 1:
                self.idx = self.idx % (self.X_train.shape[0] - 1)
            x_batch = self.X_train[indecies]
            y_batch = self.y_train[indecies]
            y_pred = self.task_model.apply(self.params, x_batch)
            loss = jnp.mean(binary_cross_entropy_loss(y_batch, y_pred))
        return loss

    def get_pred(self, parameters):
        l,b, l1, b1 = parameters[:20].reshape(2,10), parameters[20:30].reshape(10), parameters[30:40].reshape(10,1), parameters[40:].reshape(1)
        self.params["linear"]["w"] = jnp.array(l)
        self.params["linear_1"]["w"] = jnp.array(l1)
        self.params["linear"]["b"] = jnp.array(b)
        self.params["linear_1"]["b"] = jnp.array(b1)
        y_pred = self.task_model.apply(self.params, self.X_test) > 0.5
        return self.X_test, y_pred

    def get_accuracy(self, parameters, type="training"):
        l,b, l1, b1 = parameters[:20].reshape(2,10), parameters[20:30].reshape(10), parameters[30:40].reshape(10,1), parameters[40:].reshape(1)
        self.params["linear"]["w"] = jnp.array(l)
        self.params["linear_1"]["w"] = jnp.array(l1)
        self.params["linear"]["b"] = jnp.array(b)
        self.params["linear_1"]["b"] = jnp.array(b1)

        if type == "validation":
            y_pred = self.task_model.apply(self.params, self.X_val) > 0.5
            return jnp.mean(y_pred == self.y_val)
        elif type == "test":
            y_pred = self.task_model.apply(self.params, self.X_test) > 0.5
            return jnp.mean(y_pred == self.y_test)
        else:
            y_pred = self.task_model.apply(self.params, self.X_train) > 0.5
            return jnp.mean(y_pred == self.y_train)


# task = moon_task()
#
# print(task.params)
#
# b = 1
# for i in range(10000):
#     k = task.get_loss(np.random.rand(30)*5)
#     print(k)
#     if k < b:
#         print(k)
#         b = k
# print(b)