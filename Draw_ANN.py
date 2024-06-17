
############################################################################################################
############################################################################################################

### Author:  ----------> Nikolin Prenga

## Created: May 1, 2024
############################################################################################################
############################################################################################################

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotlib.
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    node_radius = 0.01
    colors = ['blue', 'green', 'coral', 'olive', 'magenta']
    nodes = []

    # Top labels for layer types
    layer_labels_top = ['Input Layer', 'First Hidden Layer', 'Second Hidden Layer', 'Third Hidden Layer', 'Output Layer']

    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        nodes.append([])

        # Add text on top of each layer
        ax.text(n*h_spacing + left, top + 0.1, layer_labels_top[n], ha='center', fontsize=20, color='black')

        # Add text below each layer
        ax.text(n*h_spacing + left, bottom - 0.1, f'Layer {n+1}', ha='center', fontsize=20, color='black')

        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), node_radius,
                                color=colors[n % len(colors)], ec='k', zorder=4)
            ax.add_artist(circle)
            nodes[-1].append(circle)

    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        for a in range(layer_size_a):
            for b in range(layer_size_b):
                line = plt.Line2D([i*h_spacing + left, (i + 1)*h_spacing + left],
                                  [nodes[i][a].center[1], nodes[i+1][b].center[1]], c='k', zorder=1)
                ax.add_artist(line)

    ax.set_xlim(left - node_radius, right + node_radius)
    ax.set_ylim(bottom - 0.2, top + 0.2)  # Adjust vertical limits to make room for text
    ax.axis('off')
    ax.set_aspect('equal')

# Define the layer sizes
layer_sizes = [18*2-3, 8*2-3, 10*2-3, 8*2-3,  4]  # Example: input layer, three hidden layers, output layer

# Create the figure
fig, ax = plt.subplots(figsize=(30, 30))
draw_neural_net(ax, .1, .9, .1, .9, layer_sizes)
plt.show()



import matplotlib.pyplot as plt

def draw_custom_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotlib, with custom labels for each node.
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))  # vertical spacing between layers
    h_spacing = (right - left) / float(len(layer_sizes) - 1)  # horizontal spacing between nodes
    node_radius = 0.07  # Increase node radius for better visibility
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey', 'magenta']

    node_centers = {}  # Dictionary to store the center positions of the nodes

    # Draw the nodes and labels
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            node_x = n * h_spacing + left
            node_y = layer_top - m * v_spacing
            circle = plt.Circle((node_x, node_y), node_radius, color=colors[n % len(colors)], ec='k', zorder=4)
            ax.add_artist(circle)
            node_centers[(n, m)] = (node_x, node_y)  # Store the center for connecting lines
            # Custom labels for each layer
            if n == 0:
                if m==layer_size-1:
                    label = f"$x_{{(n_1)}}$"
                else:
                    label = f"$x_{{({m+1})}}$"

            elif n==1:
                if m==layer_size-1:
                    label = r"$a^{(2)}_{(n_2)}=\phi\left(\sum_k  w^{(2)}_{(n_2, k)} x_{{(k)}} + b^{(2)}_{(n_2)} \right)$"
                else:
                    label = fr"$a^{{(2)}}_{{({m+1})}}=\phi\left(\sum_k  w_{{({m+1},k)}}^{{(2)}} x_{{(k)}} + b_{{({m+1})}}^{{(2)}} \right)$"


            elif n==2:
                if m==layer_size-1:

                    label = r"$a^{(3)}_{(n_3)}=\phi\left(\sum_k  w^{(3)}_{(n_3, k)} a^{(2)}_{k} + b^{(3)}_{(n_3)} \right)$"
                else:
                    label = fr"$a^{{(3)}}_{{({m+1})}}= \phi\left(\sum_k  w_{{({m+1},k)}}^{{(3)}} a_{{(k)}}^{{(2)}} + b_{{({m+1})}}^{{(3)}} \right)$"

            elif n==3:
              if m==layer_size-1:

                  label  = r"$a^{(4)}_{(n_4)}==\phi\left(\sum_k  w^{(4)}_{(n_4, k)} a^{(3)}_{n_4} + b^{(4)}_{(n_3)} \right)$"
              else:
                  label = fr"$a^{{(4)}}_{{({m+1})}}=\phi\left(\sum_k  w_{{({m+1},k)}}^{{(4)}} a_{{(k)}}^{{(3)}} + b_{{({m+1})}}^{{(4)}} \right)$"

            elif n==4:
                if m==layer_size-1:

                  label  = r"$a^{(5)}_{(n_5)}==\phi\left(\sum_k  w^{(5)}_{(n_4, k)} a^{(4)}_{k} + b^{(5)}_{(n_3)} \right)$"
                else:
                    label = fr"$a^{{(5)}}_{{({m+1})}}=\phi\left(\sum_k  w_{{({m+1},k)}}^{{(5)}} a_{{(k)}}^{{(4)}} + b_{{({m+1})}}^{{(5)}} \right)$"
                #label = f"$a_{{{m+1}}}^{{({n+1})}}$"




            ax.text(node_x, node_y, label, ha='center', va='center', fontsize=15, zorder=5)

    # Draw the edges
    for i in range(n_layers - 1):
        for a in range(layer_sizes[i]):
            for b in range(layer_sizes[i + 1]):
                line = plt.Line2D([node_centers[(i, a)][0], node_centers[(i + 1, b)][0]],
                                  [node_centers[(i, a)][1], node_centers[(i + 1, b)][1]], c='k', zorder=1)
                ax.add_artist(line)

    ax.set_xlim(left - node_radius, right + node_radius)
    ax.set_ylim(bottom - node_radius, top + node_radius)
    ax.axis('off')
    ax.set_aspect('equal')

# Define the layer sizes
layer_sizes = [5, 3, 4, 4, 2]  # Example configuration

# Create the figure
fig, ax = plt.subplots(figsize=(25, 25))
plt.title('Something')
draw_custom_neural_net(ax, .1, .9, .1, .9, layer_sizes)
plt.show()