
import jax
from jax import random, grad, jit, value_and_grad
import jax.numpy as jnp
import haiku as hk
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import jax
import optax
import jax.numpy as jnp
import haiku as hk
from flax import linen as nn
from tensorflow_datasets import as_numpy
import tensorflow_datasets as tfds
import numpy as np


# Define the model
def lenet(n_classes=10):
    def model(x):
        x = hk.Conv2D(output_channels=6, kernel_shape=5, stride=1)(x)
        x = jnp.tanh(x)
        x = hk.AvgPool(window_shape=2, strides=2,padding="VALID")(x)
        x = hk.Conv2D(output_channels=5, kernel_shape=5, stride=1)(x)
        x = jnp.tanh(x)
        x = hk.AvgPool(window_shape=3, strides=3,padding="VALID")(x)
        #x = x.reshape((x.shape[0], -1))
        x = hk.Flatten()(x)
        x = hk.Linear(120)(x)
        x = jnp.tanh(x)
        x = hk.Linear(84)(x)
        x = jnp.tanh(x)
        x = hk.Linear(n_classes)(x)
        return x
    return model

# Load MNIST dataset
def load_dataset(split, batch_size):
    ds = tfds.load('mnist', split=split, shuffle_files=True, as_supervised=True)
    ds = ds.map(lambda x, y: (x / 255, y))
    ds = ds.batch(batch_size)
    ds = as_numpy(ds)
    return ds

class mnist_task:

    def __init__(self):
        # Transforms
        self.train_dataset = load_dataset("train", batch_size=600)
        self.test_dataset = load_dataset("test", batch_size=1000)

        for i in self.train_dataset:
            self.X_train, self.y_train = i
            break
        for j in self.test_dataset:
            self.X_test, self.y_test = j
            break

        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=42)


        # Transform the model function into a pair of pure functions
        task_model = lenet(n_classes=10)
        self.task_model = hk.transform(task_model) #hk.without_apply_rng(hk.transform(task_model))


        self.optimizer = optax.adamw(0.001)
        # Initialize parameters and optimizer
        self.params = self.task_model.init(jax.random.PRNGKey(42), jnp.ones([1, 28, 28, 1]))
        self.opt_state = self.optimizer.init(self.params)

    def accuracy(self, params, images, labels):
        # Compute the logits given the images.
        logits = self.task_model.apply(params, None, images)

        # Compute the predicted classes.
        predicted_class = jnp.argmax(logits, axis=-1)

        # Check which predictions match the ground truth labels.
        correct_predictions = jnp.sum(jnp.equal(predicted_class, labels))

        # Compute the accuracy as the number of correct predictions divided by the total number of predictions.
        acc = correct_predictions / labels.size

        return acc

    def get_accuracy(self, parameters, type="training", without_slice=False):
        if not without_slice:
            c1_w, c1_b, c2_w, c2_b, l, b, l1, b1, l2, b2 = parameters[:150].reshape(5, 5, 1, 6), parameters[150:156].reshape(6),\
                                                            parameters[156:906].reshape(5, 5,6,5), parameters[906:911].reshape(5), \
                                                            parameters[911:10511].reshape(80, 120), parameters[10511:10631].reshape(120), \
                                                            parameters[10631:20711].reshape(120, 84), parameters[20711:20795].reshape(84) , \
                                                            parameters[20795:21635].reshape(84, 10), parameters[21635:].reshape(10)
            self.params["conv2_d"]["w"] = jnp.array(c1_w)
            self.params["conv2_d"]["b"] = jnp.array(c1_b)
            self.params["conv2_d_1"]["w"] = jnp.array(c2_w)
            self.params["conv2_d_1"]["b"] = jnp.array(c2_b)

            self.params["linear"]["w"] = jnp.array(l)
            self.params["linear_1"]["w"] = jnp.array(l1)
            self.params["linear_2"]["w"] = jnp.array(l2)

            self.params["linear"]["b"] = jnp.array(b)
            self.params["linear_1"]["b"] = jnp.array(b1)
            self.params["linear_2"]["b"] = jnp.array(b2)

        if type == "test":
            return self.accuracy(self.params, self.X_test, self.y_test)
        elif type == "validation":
            return self.accuracy(self.params, self.X_val, self.y_val)
        else:
            return self.accuracy(self.params, self.X_train, self.y_train)

    def loss_fn(self, params, images, labels):
        logits = self.task_model.apply(params, None, images)
        labels = jax.nn.one_hot(labels, 10)
        return jnp.mean(optax.softmax_cross_entropy(logits, labels))

    def get_loss(self, parameters, type="training", without_slice=False):
        if not without_slice:
            c1_w, c1_b, c2_w, c2_b, l, b, l1, b1, l2, b2 = parameters[:150].reshape(5, 5, 1, 6), parameters[150:156].reshape(6),\
                                                            parameters[156:906].reshape(5, 5,6,5), parameters[906:911].reshape(5), \
                                                            parameters[911:10511].reshape(80, 120), parameters[10511:10631].reshape(120), \
                                                            parameters[10631:20711].reshape(120, 84), parameters[20711:20795].reshape(84) , \
                                                            parameters[20795:21635].reshape(84, 10), parameters[21635:].reshape(10)
            self.params["conv2_d"]["w"] = jnp.array(c1_w)
            self.params["conv2_d"]["b"] = jnp.array(c1_b)
            self.params["conv2_d_1"]["w"] = jnp.array(c2_w)
            self.params["conv2_d_1"]["b"] = jnp.array(c2_b)

            self.params["linear"]["w"] = jnp.array(l)
            self.params["linear_1"]["w"] = jnp.array(l1)
            self.params["linear_2"]["w"] = jnp.array(l2)

            self.params["linear"]["b"] = jnp.array(b)
            self.params["linear_1"]["b"] = jnp.array(b1)
            self.params["linear_2"]["b"] = jnp.array(b2)

        if type == "test":
            return self.loss_fn(self.params, self.X_test, self.y_test)
        elif type == "validation":
            return self.loss_fn(self.params, self.X_val, self.y_val)
        else:
            return self.loss_fn(self.params, self.X_train, self.y_train)

    def train(self):
        pass

# task = mnist_task()
# print(task.get_accuracy(jnp.ones([21645]), "test"))