import numpy as np
import matplotlib.pyplot as plt

def visualize_layer(layer, input_feature_names=None):
    """
    Visualizes the weights and biases of a Dense Layer.

    Args:
        layer: Instance of Layer_Dense
        input_feature_names: List of strings for input names (optional)
    """
    # Extract weights and biases
    # weights shape: (n_inputs, n_neurons)
    # biases shape: (1, n_neurons)
    weights = layer.weights
    biases = layer.biases[0]

    n_inputs = weights.shape[0]
    n_neurons = weights.shape[1]

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Layout parameters
    layer_gap = 1.0

    # Calculate vertical positions to center the layers
    # We map indices to y coordinates centered around 0
    def get_y_coords(n):
        return np.linspace((n-1)/2, -(n-1)/2, n)

    input_y = get_y_coords(n_inputs)
    neuron_y = get_y_coords(n_neurons)

    # Draw weights (connections)
    max_weight = np.max(np.abs(weights)) if np.max(np.abs(weights)) > 0 else 1.0

    for i in range(n_inputs):
        for j in range(n_neurons):
            w = weights[i, j]
            # Style based on weight value
            color = 'blue' if w > 0 else 'red'
            alpha = 0.2 + 0.8 * (abs(w) / max_weight)
            width = 0.5 + 2.5 * (abs(w) / max_weight)

            # Draw line
            ax.plot([0, layer_gap], [input_y[i], neuron_y[j]],
                    c=color, alpha=alpha, linewidth=width, zorder=1)

    # Draw Input Nodes
    ax.scatter(np.zeros(n_inputs), input_y, s=400, c='white', edgecolors='black', zorder=2)
    for i in range(n_inputs):
        label = input_feature_names[i] if input_feature_names and i < len(input_feature_names) else f"In {i+1}"
        ax.text(-0.05, input_y[i], label, ha='right', va='center', fontsize=10, fontweight='bold')

    # Draw Neuron Nodes
    ax.scatter(np.ones(n_neurons) * layer_gap, neuron_y, s=400, c='white', edgecolors='black', zorder=2)
    for i in range(n_neurons):
        ax.text(layer_gap + 0.05, neuron_y[i], f"Neuron {i+1}\nBias: {biases[i]:.2f}",
                ha='left', va='center', fontsize=10)

    # Styling
    ax.set_title("Dense Layer: Weights & Biases", fontsize=14)
    ax.axis('off')
    ax.set_xlim(-0.5, layer_gap + 0.5)
    ax.set_ylim(min(input_y.min(), neuron_y.min()) - 0.5, max(input_y.max(), neuron_y.max()) + 0.5)

    plt.tight_layout()
    plt.show()
