import jax
import jax.numpy as jnp
import haiku as hk
import optax
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler





import optax
import jax
from jax import random, grad, jit, value_and_grad
import jax.numpy as jnp
import haiku as hk
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
# Define the model
def task_model(x):
    net = hk.Sequential([
        hk.Linear(20), jax.nn.relu,
        hk.Linear(7),  # There are 7 classes in the dataset
    ])
    return net(x)

def generate_data():
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
    X_train, X_test, y_train, y_test = train_test_split(data[:800,:], target[:800], test_size=0.5, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.75, random_state=42)


    # Convert to jax arrays
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    return X_train, y_train, X_test, y_test, X_val, y_val

class forest_task:

    def __init__(self, opt=None, state=42, lr=0.001):
        # Transforms
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = generate_data()
        # Transform the model function into a pair of pure functions
        self.task_model = hk.without_apply_rng(hk.transform(task_model))
        # Create optimizer
        self.optimizer = optax.noisy_sgd(lr, 0.001, 0.75)#optax.adam(0.001)
        #self.optimizer = optax.sgd(lr)
        # self.optimizer = optax.adam(lr)
        self.optimizer = optax.adamw(lr)
        #self.optimizer = opt
        # self.optimizer = optax.adagrad(0.001)
        # self.optimizer = optax.optimistic_gradient_descent(lr)
        # Initialize parameters and optimizer
        self.params = self.task_model.init(jax.random.PRNGKey(state), self.X_train[0])
        self.opt_state = self.optimizer.init(self.params)

        self.batch_size = 24
        self.idx = 0

    def get_accuracy(self, parameters, type="training", without_slice=False):
        if not without_slice:
            l, b, l1, b1 = parameters[:1080].reshape(54, 20), parameters[1080:1100].reshape(20), parameters[1100:1240].reshape(20, 7), parameters[1240:].reshape(7)
            self.params["linear"]["w"] = jnp.array(l)
            self.params["linear_1"]["w"] = jnp.array(l1)
            self.params["linear"]["b"] = jnp.array(b)
            self.params["linear_1"]["b"] = jnp.array(b1)

        if type == "test":
            logits = self.task_model.apply(self.params, self.X_test)
            pred = jnp.argmax(logits, axis=-1)
            accuracy = (pred == self.y_test).mean()
            return accuracy
        elif type == "validation":
            logits = self.task_model.apply(self.params, self.X_val)
            pred = jnp.argmax(logits, axis=-1)
            accuracy = (pred == self.y_val).mean()
            return accuracy
        else:
            logits = self.task_model.apply(self.params, self.X_train)
            pred = jnp.argmax(logits, axis=-1)
            accuracy = (pred == self.y_train).mean()
            return accuracy


    def get_loss(self, parameters, type="training", without_slice=False):
        if not without_slice:
            l, b, l1, b1 = parameters[:1080].reshape(54, 20), parameters[1080:1100].reshape(20), parameters[1100:1240].reshape(20, 7), parameters[1240:].reshape(7)
            self.params["linear"]["w"] = jnp.array(l)
            self.params["linear_1"]["w"] = jnp.array(l1)
            self.params["linear"]["b"] = jnp.array(b)
            self.params["linear_1"]["b"] = jnp.array(b1)

        if type == "test":
            logits = self.task_model.apply(self.params, self.X_test)
            return optax.softmax_cross_entropy(logits, jax.nn.one_hot(self.y_test, 7)).mean()
        if type == "validation":
            logits = self.task_model.apply(self.params, self.X_val)
            return optax.softmax_cross_entropy(logits, jax.nn.one_hot(self.y_val, 7)).mean()
        else:

            indecies = np.array(range(self.idx, self.idx + self.batch_size)) % (self.X_train.shape[0] - 1)
            self.idx += self.batch_size
            if self.idx > self.X_train.shape[0] - 1:
                self.idx = self.idx % (self.X_train.shape[0] - 1)
            #print(indecies)
            x_batch = self.X_train[indecies]
            y_batch = self.y_train[indecies]
            logits = self.task_model.apply(self.params, x_batch)
            return optax.softmax_cross_entropy(logits, jax.nn.one_hot(y_batch, 7)).mean()

            logits = self.task_model.apply(self.params, self.X_train)
            return optax.softmax_cross_entropy(logits, jax.nn.one_hot(self.y_train, 7)).mean()

    def train(self, epochs):
        @jax.jit
        def train_step(params, opt_state, x, y):
            # Compute gradients
            def loss_fn(params):
                logits = self.task_model.apply(params, x)
                loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, 7)).mean()
                return loss

            grad_fn = jax.grad(loss_fn)
            grads = grad_fn(params)

            # Update parameters
            updates, new_opt_state = self.optimizer.update(grads,opt_state,params=params)#self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)

            # grads = jax.grad(loss_fn)(params)
            #
            #
            # # Add Gaussian noise to the gradients
            # key = random.PRNGKey(0)
            # noise = jax.tree_map(lambda x: random.normal(key, x.shape), grads)
            #
            # learning_rate = 0.001
            #
            # # Stochastic gradient update with noise
            # updates = jax.tree_map(lambda g, n: -learning_rate * g + jnp.sqrt(2 * learning_rate) * 0, grads, noise)
            #
            # return jax.tree_util.tree_map(lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype),params, updates)
            return new_params, new_opt_state

        # Training loop
        vals = []
        best_val = 0
        for epoch in range(epochs):
            self.params, opt_state = train_step(self.params, self.opt_state, self.X_train, self.y_train)

            va = self.get_accuracy(self.params, "validation", True)
            vals.append(va)
            ta = self.get_accuracy(self.params, "train", True)
            ls = self.get_loss(self.params, "", True)
            # print(f"Validation: {va}")
            # print(f"Training: {ta}")
            # print(f"Loss: {ls}")
            if va > best_val:
                best_val = va

        test_acc = self.get_accuracy(self.params, "test", True)
        return test_acc, vals



            # # Evaluate on test data
            # logits = self.task_model.apply(self.params, self.X_test)
            # pred = jnp.argmax(logits, axis=-1)
            # accuracy = (pred == self.y_test).mean()
            # print(f"Test accuracy: {accuracy}")


# lrs = [0.01, 0.005, 0.0025 ,0.001, 0.0005, 0.00025]
# opt = [optax.sgd(0.005), optax.adam(0.0025), optax.noisy_sgd(0.005, 0.001, 0.75), optax.adamw(0.001)]
# vals = []
# test_accs = []
#
# for key in [42,43,44, 45, 46]:
#     temp = []
#     test = []
#     for idx, opt in enumerate(opt):
#         task = forest_task(opt, state=42)
#         test_acc, vals = task.train(10000)
#         temp.append(vals)
#         test.append(test_acc)
#
#     vals.append(temp)
#     test_accs.append(test)




#
# task = forest_task(state=42, lr=0.005)
# best_val = task.train(10000)
# print(best_val)

