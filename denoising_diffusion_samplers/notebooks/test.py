import jax
import optax
import jax.numpy as jnp
import haiku as hk
from flax import linen as nn
from sklearn.model_selection import train_test_split
from tensorflow_datasets import as_numpy
import tensorflow_datasets as tfds

# Load MNIST dataset
def load_dataset(split, batch_size):
    ds = tfds.load('mnist', split=split, shuffle_files=True, as_supervised=True)
    ds = ds.map(lambda x, y: (x / 255, y))
    ds = ds.batch(batch_size)
    ds = as_numpy(ds)
    return ds

train_dataset = load_dataset("train", batch_size=600)
test_dataset = load_dataset("test", batch_size=1000)



# Define the LeNet model
def LeNet5(n_classes):
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

# Now you can create a model instance:
lenet5 = LeNet5(n_classes=10)

# And transform it into a pair of pure functions:
net = hk.transform(lenet5)

# Make the network and loss function pure.
def loss_fn(params, images, labels):
    logits = net.apply(params, None, images)
    labels = jax.nn.one_hot(labels, 10)
    return jnp.mean(optax.softmax_cross_entropy(logits, labels))


def accuracy(params, images, labels):
    # Compute the logits given the images.
    logits = net.apply(params, None, images)

    # Compute the predicted classes.
    predicted_class = jnp.argmax(logits, axis=-1)

    # Check which predictions match the ground truth labels.
    correct_predictions = jnp.sum(jnp.equal(predicted_class, labels))

    # Compute the accuracy as the number of correct predictions divided by the total number of predictions.
    acc = correct_predictions / labels.size

    return acc


# Make the loss function pure.
#loss_fn = hk.transform(loss_fn)

# Prepare an optimizer.
opt = optax.adam(0.001)

@jax.jit
def update(params, opt_state, images, labels):
    grads = jax.grad(loss_fn)(params, images, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Initialize parameters
params = net.init(jax.random.PRNGKey(42), jnp.ones([1, 28, 28, 1]))
print(params)
opt_state = opt.init(params)


total = 0
for param in params:
    print(f"Weight shape: {params[param]['w'].shape}")
    print(f"Bias shape: {params[param]['b'].shape}")
    print(param)
    w = 1
    for i in (list(params[param]["w"].shape)):
        w *= i
    b = 1
    for j in list(params[param]["b"].shape):
        b *= j
    total += w
    total += b

print(f"Total: {total}")# print("Total parameters: ", count_parameters(params))

# for epoch in range(1000):
#     for batch in train_dataset:
#         images, labels = batch
#         images = jnp.array(images)
#         labels = jnp.array(labels)
#         params, opt_state = update(params, opt_state, images, labels)
#         print(accuracy(params, images, labels))
#         break
#
# for b in test_dataset:
#     test_images, test_labels = b
#     print(accuracy(params, test_images, test_labels))

