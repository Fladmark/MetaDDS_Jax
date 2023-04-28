import pickle
import matplotlib.pyplot as plt

# Your array
data_array = [1, 2, 3, 4, 5]

# Save the array to a pickle file
def save_array_to_pickle(array, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(array, file)

def load_array_from_pickle(file_name):
    with open(file_name, 'rb') as file:
        loaded_array = pickle.load(file)
    return loaded_array


# Plot the training loss on a graph
def plot_training_loss(loss_array, save_name=None):
    plt.plot(loss_array)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.grid()
    plt.show()
    if save_name:
        plt.savefig("notebooks/div_files/" + save_name + ".png")

def plot_training_and_validation_losses(training_loss_array, validation_loss_array):
    plt.plot(training_loss_array, label='Training Loss')
    plt.plot(validation_loss_array, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses vs. Epochs')
    plt.grid()
    plt.legend()
    plt.show()

def get_smallest_loss(augmented_trajectory, config):
    data_x = []
    for sample in augmented_trajectory:
        x = sample[-1][:config.model.input_dim]
        data_x.append(x)
    b = 1
    w = None
    for weights in data_x:
        l = config.model.target_class.f(weights)
        if l < b:
            b = l
            w = weights
    return b, w

def get_smallest_validation_loss(augmented_trajectory, config):
    data_x = []
    for sample in augmented_trajectory:
        x = sample[-1][:config.model.input_dim]
        data_x.append(x)
    b = 1
    w = None
    for weights in data_x:
        l = config.model.target_class.f_val(weights)
        if l < b:
            b = l
            w = weights
    return b, w