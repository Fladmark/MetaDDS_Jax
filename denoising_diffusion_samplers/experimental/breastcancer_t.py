# import numpy as np
# import jax
# import jax.numpy as jnp
# from jax import random, jit, grad, value_and_grad
# from sklearn.datasets import make_moons, make_circles, load_breast_cancer, load_wine
# from sklearn.model_selection import train_test_split
# import haiku as hk
# import optax
#
# def model_fn(x):
#     net = hk.Sequential([
#         hk.Linear(4), jax.nn.relu,
#         hk.Linear(1), jax.nn.sigmoid
#     ])
#     return net(x)
#
# # def loss_fn(params, x, y_true, model):
# #     y_pred = model.apply(params, None, x)
# #     return jnp.mean((y_true - y_pred) ** 2)
#
# def loss_fn(params, x, y_true, model):
#     y_pred = model.apply(params, x)
#     epsilon = 1e-7
#     y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
#     return -(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))
#
# def accuracy(params, x, y_true, model):
#     y_pred = model.apply(params, None, x) > 0.5
#     return jnp.mean(y_pred == y_true)
#
# # Define the update function
# @jit
# def update(params, x, y_true, opt_state, optimizer):
#     grads = jax.grad(loss_fn)(params, x, y_true)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state
#
# class breast_task:
#
#     def __init__(self):
#         X, y = load_breast_cancer(return_X_y=True)
#         X = jnp.array(X[:, :15], dtype=jnp.float32)
#         y = jnp.array(jnp.expand_dims(y, 1), dtype=jnp.float32)
#         self.X_train, X_test, self.y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#         self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=422)
#
#         # Define the model
#
#         # Initialize the model
#         self.key = random.PRNGKey(43)
#         self.sample_input = jnp.array(self.X_train[0])
#         #self.model = hk.transform(model_fn)
#         self.model = hk.without_apply_rng(hk.transform(model_fn))
#         self.params = self.model.init(self.key, self.sample_input)
#
#     def get_loss(self, parameters, type="training"):
#         l, l1 = parameters[:60].reshape(15,4), parameters[60:].reshape(4,1)
#         self.params["linear"]["w"] = l
#         self.params["linear_1"]["w"] = l1
#         if type == "validation":
#             return loss_fn(self.params, self.X_val, self.y_val, self.model)
#         elif type == "test":
#             return loss_fn(self.params, self.X_test, self.y_test, self.model)
#         else:
#             return loss_fn(self.params, self.X_train, self.y_train, self.model)
#
#     def get_val_loss(self, parameters):
#         l, l1 = parameters[:60].reshape(15,4), parameters[60:].reshape(4,1)
#         self.params["linear"]["w"] = l
#         self.params["linear_1"]["w"] = l1
#         return loss_fn(self.params, self.X_val, self.y_val, self.model)
#
#     def get_test_accuracy(self, parameters):
#         print(parameters)
#         l, l1 = parameters[:60].reshape(15,4), parameters[60:].reshape(4,1)
#         self.params["linear"]["w"] = l
#         self.params["linear_1"]["w"] = l1
#         print(self.params)
#         return accuracy(self.params, self.X_test, self.y_test, self.model)
#
#     def set_weight(self, parameters):
#         l, l1 = parameters[:60].reshape(15,4), parameters[60:].reshape(4,1)
#         self.params["linear"]["w"] = l
#         self.params["linear_1"]["w"] = l1
#
#     def reset_weight(self, key_number=None):
#         if key_number:
#             self.params = self.model.init(random.PRNGKey(key_number), self.sample_input)
#         else:
#             self.params = self.model.init(self.key, self.sample_input)
#
#     def fine_tune(self, params):
#         l, l1 = params[:60].reshape(15,4).astype(jnp.float32), params[60:].reshape(4,1).astype(jnp.float32)
#
#         def a(shape, dtype):
#             return l
#         def b(shape, dtype):
#             return l1
#
#         def model_updated(x):
#             net = hk.Sequential([
#                 hk.Linear(4, w_init=a), jax.nn.relu,
#                 hk.Linear(1, w_init=b), jax.nn.sigmoid
#             ])
#             return net(x)
#
#         self.model = hk.transform(model_updated)
#         params = self.model.init(self.key, self.sample_input)
#
#         optimizer = optax.adam(0.0001)
#         opt_state = optimizer.init(params)
#
#         # Train the model
#         epochs = 10000
#         for epoch in range(epochs):
#             grads = jax.grad(loss_fn)(params, self.X_train,  self.y_train, self.model)
#             updates, opt_state = optimizer.update(grads, opt_state)
#             params = optax.apply_updates(params, updates)
#             #params, opt_state = update(params, self.X_train, self.y_train, opt_state)
#             if epoch % 100 == 0:
#                 train_loss = loss_fn(params, self.X_train, self.y_train, self.model)
#                 print(f"Epoch: {epoch}, Loss: {train_loss}")
#
#         # Evaluate the model
#
#         train_acc = accuracy(params, jnp.array(self.X_train), jnp.array(self.y_train), self.model)
#         test_acc = accuracy(params, jnp.array(self.X_test), jnp.array(self.y_test), self.model)
#         print(f"Train accuracy: {train_acc}, Test accuracy: {test_acc}")
#
#         return params


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
from sklearn.datasets import load_breast_cancer
import haiku as hk
import optax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the simple neural network
def task_model(x):
    HIDDENSIZE = 10
    return hk.Sequential([
        hk.Linear(HIDDENSIZE),
        jax.nn.relu,
        hk.Linear(1),
        jax.nn.sigmoid
    ])(x)

def generate_breast_cancer_data():
    data = load_breast_cancer()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y.reshape(-1, 1)

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = jnp.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

def accuracy(params, x, y_true, model):
    y_pred = model.apply(params, x) > 0.5
    return jnp.mean(y_pred == y_true)

class breast_task:

    def __init__(self, state=42):
        # Transforms
        self.task_model = hk.without_apply_rng(hk.transform(task_model))
        self.X, self.y = generate_breast_cancer_data()
        self.X = self.X[:, :15]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=42)

        self.idx = 0
        self.batch_size = 24
        # Initialize parameters
        DATASIZE = self.X.shape[1]
        GTSIZE = 1
        key = random.PRNGKey(state)
        x_sample = jnp.zeros((1, DATASIZE))
        self.params = self.task_model.init(key, x_sample)
        # for p in self.params:
        #     print(self.params[p]["w"].shape)
        #     print(self.params[p]["b"].shape)

    def get_loss(self, parameters, type="training"):
        l, b, l1, b1 = parameters[:150].reshape(15, 10), parameters[150:160].reshape(10), parameters[160:170].reshape(10, 1), parameters[170:].reshape(1)
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
        else:
            y_pred = self.task_model.apply(self.params, self.X_train)
            loss = jnp.mean(binary_cross_entropy_loss(self.y_train, y_pred))
            # indecies = np.array(range(self.idx, self.idx + self.batch_size)) % (self.X_train.shape[0] - 1)
            # self.idx += self.batch_size
            # if self.idx > self.X.shape[0] - 1:
            #     self.idx = self.idx % (self.X_train.shape[0] - 1)
            # #print(indecies)
            # x_batch = self.X_train[indecies]
            # y_batch = self.y_train[indecies]
            # y_pred = self.task_model.apply(self.params, x_batch)
            # loss = jnp.mean(binary_cross_entropy_loss(y_batch, y_pred))
        return loss

    def get_training_loss(self, parameters, type="training"):
        y_pred = self.task_model.apply(parameters, self.X_train)
        loss = jnp.mean(binary_cross_entropy_loss(self.y_train, y_pred))
        return loss

    def get_test_loss(self, parameters, type="training"):
        y_pred = self.task_model.apply(parameters, self.X_test)
        loss = jnp.mean(binary_cross_entropy_loss(self.y_test, y_pred))
        return loss

    def get_test_accuracy(self, parameters):
        l, b, l1, b1 = parameters[:150].reshape(15, 10), parameters[150:160].reshape(10), parameters[160:170].reshape(10, 1), parameters[170:].reshape(1)
        self.params["linear"]["w"] = jnp.array(l)
        self.params["linear_1"]["w"] = jnp.array(l1)
        self.params["linear"]["b"] = jnp.array(b)
        self.params["linear_1"]["b"] = jnp.array(b1)
        return accuracy(self.params, self.X_test, self.y_test, self.task_model)

    def get_accuracy(self, parameters, type="training"):
        l, b, l1, b1 = parameters[:150].reshape(15, 10), parameters[150:160].reshape(10), parameters[160:170].reshape(10, 1), parameters[170:].reshape(1)
        self.params["linear"]["w"] = jnp.array(l)
        self.params["linear_1"]["w"] = jnp.array(l1)
        self.params["linear"]["b"] = jnp.array(b)
        self.params["linear_1"]["b"] = jnp.array(b1)
        if type == "test":
            return accuracy(self.params, self.X_test, self.y_test, self.task_model)
        elif type == "validation":
            return accuracy(self.params, self.X_val, self.y_val, self.task_model)
        else:
            return accuracy(self.params, self.X_train, self.y_train, self.task_model)

    def train(self, opt, epochs, learning_rate=0.005):
        # Setup optimizer
        optimizer = opt#optax.noisy_sgd(learning_rate, 0.001, 0.75)
        opt_state = optimizer.init(self.params)


        @jit
        def update(params, opt_state):
            grads = jax.grad(self.get_training_loss)(params, "training")
            updates, opt_state = optimizer.update(grads, opt_state,params=params)
            params = optax.apply_updates(params, updates)
            #params = opt.update(grads, params)
            return params, opt_state

        # Training loop
        vals = []
        for epoch in range(epochs):
            # Update parameters
            self.params, opt_state = update(self.params, opt_state)
            vals.append(accuracy(self.params, self.X_val, self.y_val, self.task_model))

        # Compute loss and accuracy
        train_loss = self.get_training_loss(self.params, "training")
        train_accuracy = accuracy(self.params, self.X_train, self.y_train, self.task_model)
        test_loss = self.get_test_loss(self.params, "test")
        test_accuracy = accuracy(self.params, self.X_test, self.y_test, self.task_model)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        return accuracy(self.params, self.X_test, self.y_test, self.task_model), vals



# lrs = [0.01, 0.005, 0.0025 ,0.001, 0.0005, 0.00025]
#
# vals = [0] * 6
# #for key in [42,43,44]:
# for idx, lr in enumerate(lrs):
#     task = breast_task()
#     best_val = task.train(epochs=20000, learning_rate=lr)
#     vals[idx] += best_val
#
# print(vals)