import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import *

def draw_true_world(data, cmap='viridis'):    
    classes = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=classes, cmap=cmap, edgecolor='k')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Points by Class')
    plt.show()

def draw_observed_world(data, cmap='viridis'):
    x = data[:, 3]
    y = data[:, 5]
    classes = data[:, 0]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=classes, cmap=cmap, edgecolor='k')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Points by Class')
    plt.show()


def draw_combined_world(data, sample_data = None, cmap='viridis'):
    
    # Extracting data for the true world
    true_classes = data[:, 0]
    true_f1 = data[:, 1]
    true_f2 = data[:, 2]
    
    # Extracting data for the observed worlds
    obs_f1 = data[:, 3]
    obs_f2 = data[:, 5]
    
    if sample_data is not None:
        obs_samp_classes = sample_data[:, 0]
        obs_samp_f1 = sample_data[:, 3]
        obs_samp_f2 = sample_data[:, 5]
    
    # Creating subplots
    fig, axs = plt.subplots(1, 3 if sample_data is not None else 2, figsize=(24 if sample_data is not None else 16, 8))
    
    # True World
    axs[0].scatter(true_f1, true_f2, c=true_classes, cmap=cmap, edgecolor='k')
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    axs[0].set_title('True World')
    
    # Observed World
    axs[1].scatter(obs_f1, obs_f2, c=true_classes, cmap=cmap, edgecolor='k')
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')
    axs[1].set_title('Observed World')
    
    if sample_data is not None:
        # Observed World Sampled
        axs[2].scatter(obs_samp_f1, obs_samp_f2, c=obs_samp_classes, cmap=cmap, edgecolor='k')
        axs[2].set_xlabel('Feature 1')
        axs[2].set_ylabel('Feature 2')
        axs[2].set_title('Observed World Sampled')
    
    # Display the plot
    plt.show()

def draw_world_grid(world_coords, feature_bounds):
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(111)

    plt.title("Rectangles")
    for (_x1, _x2), (_y1, _y2) in world_coords:
        rect1 = Rectangle((_x1, _y1), abs(_x2 - _x1), abs(_y2 - _y1), color = 'orange', fc = 'none', lw = 2)
        ax.add_patch(rect1)

    plt.xlim(feature_bounds[0])
    plt.ylim(feature_bounds[1])

    plt.show()

def plot_combined_3d(world, cmap='viridis'):
    # Extracting data for the first 3D plot
    classes_3d_1 = world[:, 0]
    f1_1 = world[:, 1]
    f2_1 = world[:, 2]
    f3_1 = world[:, 3]

    # Extracting data for the second 3D plot
    classes_3d_2 = world[:, 0]
    f1_2 = world[:, 4]
    f2_2 = world[:, 6]
    f3_2 = world[:, 8]

    # Creating subplots
    fig = plt.figure(figsize=(20, 8))

    # First 3D Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(f1_1, f2_1, f3_1, c=classes_3d_1, cmap=cmap, edgecolor='k')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.set_title('True World')

    # Second 3D Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(f1_2, f2_2, f3_2, c=classes_3d_2, cmap=cmap, edgecolor='k')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_zlabel('Feature 3')
    ax2.set_title('Observed world')

    # Display the combined plots
    plt.show()
    
def plot_3d_world(world, cmap='viridis'):
    
    # Extracting data for the 3D plot
    classes_3d = world[:, 0]
    f1 = world[:, 1]
    f2 = world[:, 2]
    f3 = world[:, 3]

    # Creating subplots with different viewing angles
    fig = plt.figure(figsize=(20, 8))

    # First 3D Plot - Default view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(f1, f2, f3, c=classes_3d, cmap=cmap, edgecolor='k')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.set_title('True World - Default View')

    # Second 3D Plot - Different view angle
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(f1, f2, f3, c=classes_3d, cmap=cmap, edgecolor='k')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_zlabel('Feature 3')
    ax2.set_title('True World - Rotated View')
    ax2.view_init(elev=30., azim=120)  # Changing the elevation and azimuth

    # Display the combined plots
    plt.show()

def draw_world_from_seed(seed, features, NUMBER_OF_CLASSES, endT = .2, endA = .3, max_depth = 4, feature_bounds = (0, 1)):
    world = generate_world(seed, features, 
                           feature_bounds = feature_bounds, 
                           endT = endT, endA = endA, max_depth = max_depth,
                           class_number = NUMBER_OF_CLASSES,
                           errRange = 1, corr = 1)
    if features == 3:
        plot_3d_world(world)
    else:
        draw_true_world(world)