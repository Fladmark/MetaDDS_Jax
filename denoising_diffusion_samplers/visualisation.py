import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def graph_2d(dict, save_name=None, sde=True):
    if sde:
        aug = "sde"
        samples = dict[-1]["aug"]
    else:
        aug = "ode"
        samples = dict[-1]["aug_ode"]

    data = []
    for sample in samples:
        s = sample[-1][0]
        data.append(s)

    # Compute the histogram values
    hist, bin_edges = np.histogram(data, bins=50)

    # Compute the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create a line plot using Seaborn
    sns.lineplot(x=bin_centers, y=hist)

    # Set the x and y axis labels
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    if save_name:
        plt.savefig(f"images/" + save_name + f"_graph_2d_{aug}.png")

    # Show the plot
    plt.show()


def hist_3d(dict, save_name=None, sde=True):
    if sde:
        aug = "sde"
        samples = dict[-1]["aug"]
    else:
        aug = "ode"
        samples = dict[-1]["aug_ode"]
    data_x = []
    data_y = []
    for sample in samples:
        x = sample[-1][0]
        y = sample[-1][1]
        data_x.append(x)
        data_y.append(y)

    # Compute the 2D histogram values
    hist, x_edges, y_edges = np.histogram2d(data_x, data_y, bins=50)

    # Create the x, y grid (mesh) for the 3D plot
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a 3D bar plot
    ax.bar3d(x_mesh.ravel(), y_mesh.ravel(), np.zeros_like(hist).ravel(), dx=0.8 * (x_edges[1] - x_edges[0]),
             dy=0.8 * (y_edges[1] - y_edges[0]), dz=hist.ravel())

    # Set the x, y, and z axis labels
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    ax.set_zlabel('Frequency')

    if save_name:
        plt.savefig("images/" + save_name + f"_hist_3d_{aug}.png")

    # Show the plot
    plt.show()

def plot_3d(dict, save_name=None, sde=True):
    if sde:
        aug = "sde"
        samples = dict[-1]["aug"]
    else:
        aug = "ode"
        samples = dict[-1]["aug_ode"]
    data_x = []
    data_y = []
    data_z = []
    for sample in samples:
        x = sample[-1][0]
        y = sample[-1][1]
        z = abs(sample[-1][2])
        data_x.append(x)
        data_y.append(y)
        data_z.append(z)

    # Convert the data lists to numpy arrays
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_z = np.array(data_z)

    # Normalize the z values for colormap
    z_norm = (data_z - data_z.min()) / (data_z.max() - data_z.min())

    # Create the 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data_x, data_y, data_z, c=z_norm, cmap='viridis')

    # Set the x, y, and z axis labels
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    ax.set_zlabel('Z Value')

    # Add a colorbar to show the intensity scale
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Heat (Z Value)')

    if save_name:
        plt.savefig("images/" + save_name + f"_plot_3d_{aug}.png")

    # Show the plot
    plt.show()

def heat_2d(dict, save_name=None, sde=True):
    if sde:
        aug = "sde"
        samples = dict[-1]["aug"]
    else:
        aug = "ode"
        samples = dict[-1]["aug_ode"]
    data_x = []
    data_y = []
    for sample in samples:
        x = sample[-1][0]
        y = sample[-1][1] # change when 3d
        data_x.append(x)
        data_y.append(y)

    # Compute the 2D histogram values
    hist, x_edges, y_edges = np.histogram2d(data_x, data_y, bins=50)

    # Create the 2D heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(hist.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto',
                   interpolation='nearest')

    # Set the x, y axis labels
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')

    # Add a colorbar to show the intensity scale
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Frequency')

    if save_name:
        plt.savefig("images/" + save_name + f"_heat_2d_{aug}.png")

    # Show the plot
    plt.show()

def plot_2d(dict, save_name=None, sde=True):
    if sde:
        aug = "sde"
        samples = dict[-1]["aug"]
    else:
        aug = "ode"
        samples = dict[-1]["aug_ode"]
    data_x = []
    data_y = []
    data_z = []
    for sample in samples:
        x = sample[-1][0]
        y = sample[-1][1]
        z = abs(sample[-1][2])
        data_x.append(x)
        data_y.append(y)
        data_z.append(z)

    def create_heatmap(x, y, z):
        # Create a 2D histogram from the data
        heatmap, xedges, yedges = np.histogram2d(x, y, weights=z, bins=50)

        # Normalize the heatmap
        heatmap = heatmap / np.sum(heatmap)

        # Plot the heatmap
        plt.imshow(heatmap, origin='lower', cmap='viridis', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.colorbar(label='Expected Value')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2D Heatmap')

        if save_name:
            plt.savefig("images/" + save_name + f"_plot_2d_{aug}.png")

        plt.show()

    create_heatmap(data_x, data_y, data_z)