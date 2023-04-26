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
def plot_training_loss(loss_array):
    plt.plot(loss_array)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.grid()
    plt.show()