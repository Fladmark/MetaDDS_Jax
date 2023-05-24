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

train_dataset = load_dataset("train", batch_size=300)
test_dataset = load_dataset("test", batch_size=500)



# Define the LeNet model
def LeNet5(n_classes):
    def model(x):
        x = hk.Conv2D(output_channels=4, kernel_shape=5, stride=1)(x)
        x = jnp.tanh(x)
        x = hk.AvgPool(window_shape=2, strides=2,padding="VALID")(x)
        x = hk.Conv2D(output_channels=3, kernel_shape=5, stride=1)(x)
        x = jnp.tanh(x)
        x = hk.AvgPool(window_shape=3, strides=3,padding="VALID")(x)
        #x = x.reshape((x.shape[0], -1))
        x = hk.Flatten()(x)
        x = hk.Linear(40)(x)
        x = jnp.tanh(x)
        x = hk.Linear(20)(x)
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
    print(total)
    total += b
    print(total)

print(f"Total: {total}")# print("Total parameters: ", count_parameters(params))

for j in test_dataset:
    X_test, y_test = j
    break

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5,
                                                                    random_state=42)

for epoch in range(3000):
    for batch in train_dataset:
        images, labels = batch
        images = jnp.array(images)
        labels = jnp.array(labels)
        params, opt_state = update(params, opt_state, images, labels)
        print(accuracy(params, X_val, y_val))
        break


print("Test")
print(accuracy(params, X_test, y_test))

# iterator = iter(train_dataset)
# while True:
#     try:
#         self.X_train, self.y_train = (next(iterator))
#     except:
#         iterator = iter(train_dataset)
#         self.X_train, self.y_train = (next(iterator))