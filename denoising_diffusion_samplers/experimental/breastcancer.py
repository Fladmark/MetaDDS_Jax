import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, grad, value_and_grad
from sklearn.datasets import make_moons, make_circles, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import haiku as hk
import optax
import random as r
from flax.core import freeze, unfreeze

# # Create the make_moons dataset
#X, y = make_circles(n_samples=2000, noise=0.1, factor=0.5)
# #X, y = make_moons(n_samples=1000, noise=0.1)
X, y = load_breast_cancer(return_X_y=True)

print(X[:, :15].shape)

print(X.shape)
print(y.shape)

X = jnp.array(X[:, :15], dtype=jnp.float32)
y = jnp.array(jnp.expand_dims(y, 1), dtype=jnp.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=422)



# Define the model
def model_fn(x):
    net = hk.Sequential([
        hk.Linear(4), jax.nn.relu,
        hk.Linear(1), jax.nn.sigmoid
    ])
    return net(x)

# Initialize the model
key = random.PRNGKey(42)
sample_input = jnp.array(X_train[0])
model = hk.transform(model_fn)
params = model.init(key, sample_input)

# print(params)
#
# k = jnp.array(np.random.random_sample(size = 64))
# z = jnp.array(np.zeros(4))
# y = jnp.zeros(1)

# new_params = {"linear": {
#     "w": k[:60].reshape(15, 4).astype(jnp.float32),
#     "b": z.astype(jnp.float32)
# },
# "linear_1": {
#     "w": k[60:].reshape(4, 1).astype(jnp.float32),
#     "b": y.astype(jnp.float32)
#     }
# }
#
# def model_fn2(x):
#     net = hk.Sequential([
#         hk.Linear(4, w_init=l), jax.nn.relu,
#         hk.Linear(1, w_init=l1), jax.nn.sigmoid
#     ])
#     return net(x)
#
# model = hk.transform(model_fn)
# params = model.init(key, sample_input)
#
# print(params)

def loss_fn(params, x, y_true, m):
    y_pred = m.apply(params, None, x)
    return jnp.mean((y_true - y_pred) ** 2)

print(loss_fn(params, X_train, y_train, model))
k = np.array(np.random.random_sample(size = 64))
k = np.linspace(-3, 3.0, num=64)
h = jnp.array(np.ones(64))
l, l1 = jnp.array(np.array(k[:60].reshape(15, 4)).astype(jnp.float32).tolist()), jnp.array(np.array(k[60:].reshape(4, 1)).astype(jnp.float32).tolist())

print(type(l))
print(type(l[0]))
print(type(l[0][0]))


#l = jnp.array([[0.6678470756127151, 0.3274291490304583, 0.30266900889984805, 0.21376632417946362], [0.5103366641606529, 0.9436193439647486, 0.04727112750652285, 0.774629318134999], [0.8223630717785946, 0.9784832666339504, 0.5521222094695053, 0.9434604113935086], [0.7181365165985387, 0.6609520539467061, 0.38629393464153083, 0.4995125039994298], [0.0026219095061131847, 0.24354183266535645, 0.8896728471450788, 0.19232991891444262], [0.644857176184392, 0.01696018933465948, 0.347170099067148, 0.3108150336622806], [0.09052074821370448, 0.7513895304155646, 0.7939472902095046, 0.5140052699457585], [0.3650721209950857, 0.07853242210753486, 0.5336413896163417, 0.19239249944867032], [0.6366474950602764, 0.195258352738363, 0.3545856904979694, 0.36619585586516434], [0.7304597413046711, 0.09482598452317426, 0.776057785812247, 0.8205379984661174], [0.9845696647941284, 0.7090913628836085, 0.6020420484295513, 0.6465266539428827], [0.42107359793092203, 0.5633387365403031, 0.34591901328966757, 0.06921357945928963], [0.9711444361920051, 0.5368569769437099, 0.09283313233432589, 0.6311622061483501], [0.5290310357578624, 0.008487054184346698, 0.7077727853850005, 0.09212484211065186], [0.4167362228590241, 0.1023526989115221, 0.954863477769642, 0.8555861363596458]])
# l = jnp.array([[ 0.2187946 , -0.23228885,  0.08327011, -0.20316881],
#        [ 0.31600928,  0.38450536, -0.4181966 ,  0.0998612 ],
#        [ 0.15788722, -0.0975444 , -0.09512267, -0.33138397],
#        [-0.46265787, -0.25270176, -0.32220381,  0.26525167],
#        [ 0.19369553,  0.08855311,  0.17170094,  0.01689583],
#        [-0.07537406,  0.1501624 , -0.30576172,  0.18092866],
#        [-0.17359711, -0.18658705, -0.22045238, -0.14906436],
#        [ 0.13163392,  0.40099695,  0.12509108,  0.31128183],
#        [-0.02491019,  0.09123949,  0.13589908, -0.15668334],
#        [-0.50324094,  0.00210878, -0.31612965,  0.14466567],
#        [ 0.38123327, -0.4001476 , -0.16298544,  0.4324488 ],
#        [ 0.07594585, -0.06827579,  0.05286605,  0.06222933],
#        [ 0.04543871,  0.2116714 ,  0.33527023, -0.22588463],
#        [-0.03270659, -0.40571335,  0.0664184 ,  0.15220305],
#        [ 0.37229028,  0.15730236,  0.18755937,  0.3549387 ]])

# l = jnp.array([[ 1 , 1,  1, 1],
#        [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],
#                [ 1 , 1,  1, 1],])
#
# l1 = jnp.array([[ 0.32167414],
#        [-0.05926372],
#        [ 0.44483188],
#        [-0.16993079]])

print(type(l1[0][0]))

def a(shape, dtype):
    global l
    return l


def b(shape, dtype):
    global l1
    return l1


def model_updated(x):
    net = hk.Sequential([
        hk.Linear(4, w_init=a), jax.nn.relu,
        hk.Linear(1, w_init=b), jax.nn.sigmoid
    ])
    return net(x)


model2 = hk.transform(model_updated)
params2 = model2.init(key, sample_input)

print(loss_fn(params2, X_train, y_train, model2))
exit()


# b = 1
# w = None
# for i in range(5000):
#     key = random.PRNGKey(i)
#     model = hk.transform(model_fn)
#     params = model.init(key, sample_input)
#     l = loss_fn(params, X_train, y_train)
#     if l < b:
#         w = params
#         b = l
#
# params = w



# Create the optimizer
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)

# Define the update function
@jit
def update(params, x, y_true, opt_state):
    grads = jax.grad(loss_fn)(params, x, y_true)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Train the model
epochs = 10000
for epoch in range(epochs):
    params, opt_state = update(params, X_train, y_train, opt_state)
    if epoch % 100 == 0:
        train_loss = loss_fn(params, X_train, y_train)
        print(f"Epoch: {epoch}, Loss: {train_loss}")

# Evaluate the model
def accuracy(params, x, y_true):
    y_pred = model.apply(params, None, x) > 0.5
    return jnp.mean(y_pred == y_true)

train_acc = accuracy(params, jnp.array(X_train), jnp.array(y_train))
test_acc = accuracy(params, jnp.array(X_test), jnp.array(y_test))
print(f"Train accuracy: {train_acc}, Test accuracy: {test_acc}")

train_loss = loss_fn(params, jnp.array(X_train), jnp.array(y_train))
test_loss = loss_fn(params, jnp.array(X_test), jnp.array(y_test))
print(f"Train loss: {train_loss}, Test loss: {test_loss}")