import numpy as np
import torch

labels = {}
models = {}

"""
Your model specifications
"""
class Model(torch.nn.Module):

    def __init__(self):
        return 0

    def forward(self, x):
        return 0

"""
Assume this is your already defined training/test data with 155 
length rows, and the respective cost lists
"""
training_data = []
testing_data = []
cost_list_train = []
cost_list_test = []


"""
Get combinations
"""
combinations = []
for i in range(11):
    for j in range(i+1, 11):
        combinations.append((i, j))


"""
Setup dictionaries
"""
for pair in combinations:
    models[pair] = Model()
    labels[pair] = []


"""
Get training labels
"""
for idx, row in enumerate(training_data):
    for pair in combinations:
        label = np.argmin(np.array([cost_list_train[idx][pair[0]], cost_list_train[idx][pair[1]]]))
        labels[pair].append(label)

"""
Training loop
"""
for pair in combinations:
    current_model = models[pair]
    current_labels = labels[pair]

    # Do standard training from here


"""
We now assume the models dictionary has all fully trained models.
Thus, we move to the evaluation stage.
"""
predictions = []
for idx, row in enumerate(testing_data):
    decision_vector = [0]*11
    for pair in combinations:
        current_model = models[pair]
        result = current_model(row) # Get prediction of pair-model on data (will be 0 or 1)
        decision_vector[pair[result]] += 1 # Use the returned value to index the pair, to then increment the decision vector

    decision = np.argmax(decision_vector)
    predictions.append(predictions)

"""
Lastly, you can calculate accuracy by comparing predictions with cost_list_test
"""
num_correct = sum([1 for i in range(len(predictions)) if predictions[i] == cost_list_test[i]])
accuracy = num_correct / len(predictions)

