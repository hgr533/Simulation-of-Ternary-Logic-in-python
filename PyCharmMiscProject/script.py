import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulate ternary logic operations for a ternary AI chip
# Ternary values: -1 (negative), 0 (neutral), +1 (positive)

# ----------------------------
# Ternary logic gates
# ----------------------------
def ternary_and(a, b):
    """Ternary AND: Returns min of two ternary values {-1, 0, 1}"""
    return np.min(a, b)


def ternary_or(a, b):
    """Ternary OR: Returns max of two ternary values {-1, 0, 1}"""
    return np.max(a, b)


def ternary_not(a):
    """Ternary NOT: Inverts -1 to +1, +1 to -1, 0 stays 0"""
    return -a

# ----------------------------
# Neural operations
# ----------------------------
def ternary_mac(inputs, weights, bias=0, t_low=-1, t_high=1):
    """
    Ternary Multiply-Accumulate with thresholds
    inputs, weights ∈ {-1,0,+1}
    bias is an integer offset
    thresholds (t_low, t_high) define mapping back to {-1,0,+1}
    """
    result = np.dot(inputs, weights) + bias
    # Clip result to ternary values {-1, 0, +1}
    if result >= t_high:
        return 1
    elif result <= t_low:
        return -1
    else:
        return 0

# Example: Simulate a ternary neural network layer
def ternary_layer(inputs, weight_matrix, biases=None, t_low=-1, t_high=1):

    """Simulate a single ternary neural network layer
    (NN layer)
    weight_matrix: shape (num_neurons, num_inputs)
    biases: optional array of biases
    """
    if biases is None:
        biases = np.zeros_like(weight_matrix.shape[0], dtype=int)
    outputs = [ternary_mac(inputs, weight_matrix[i], biases[i], t_low, t_high)
               for i in range(weight_matrix.shape[0])]
    return np.array(outputs)

# ----------------------------
# Multi-layer network with trace + animation
# ----------------------------
class TernaryNN:
    def __init__(self, layer_configs, t_low=-1, t_high=1):
        """
        layer_configs: list of dicts, each with:
            - "weights": 2D numpy array (num_neurons x num_inputs)
            - "biases": 1D numpy array (optional)
        """
        self.layers = layer_configs
        self.t_low = t_low
        self.t_high = t_high

    def forward(self, inputs, trace=False):
        """Run inputs through all layers sequentially"""
        activations = inputs
        all_activations = [activations.copy()] # store input as layer 0

        if trace:
            print("Input:", activations)

        for idx, layer in enumerate(self.layers, 1):
            W = layer["weights"]
            b = layer.get("biases", None)
            activations = ternary_layer(activations, W, b, self.t_low, self.t_high)
            all_activations.append(activations.copy())
            if trace:
                print(f" Layer {idx} activations: {activations}")

        return activations, all_activations

    def animate(self, all_activations, interval=1000):

        """
        Animate activations as a heatmap, updating layer by layer.
        interval: time per frame in ms
        """
        max_neurons = max(len(act) for act in all_activations)
        num_layers = len(all_activations)

        # Pad activations into matrix
        activ_matrix = np.full((max_neurons, num_layers), np.nan)
        for i, act in enumerate(all_activations):
            activ_matrix[:len(act), i] = act

        fig, ax = plt.subplots(figsize=(6, 4))
        cmap = plt.cm.get_cmap("RdYlGn",3)
        cax = ax.imshow(np.full_like(activ_matrix, np.nan), cmap=cmap,
                        vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(num_layers))
        ax.set_xticklabels([f"Layer {i}" for i in range(num_layers)])
        ax.set_xticks(range(max_neurons))
        ax.set_xticklabels([f"N{j}" for j in range(max_neurons)])
        plt.colorbar(cax, ticks=[-1, 0, 1], label="Ternary value")
        plt.title("Ternary NN Activations (animated)")
        plt.xlabel("Layers")
        plt.ylabel("Neurons")

        # Update function for animation
        def update(frame):
            partial_matrix = np.full_like(activ_matrix, np.nan)
            partial_matrix[:, :frame + 1] = activ_matrix[:, :frame + 1]
            cax.set_data(partial_matrix)
            return [cax]

        ani = animation.FuncAnimation(fig, update, frames=num_layers,
                                      interval=interval, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()

# ----------------------------
# Example usage
# ----------------------------
# Test the ternary operations
if __name__ == "__main__":
    # Example inputs and weights (ternary values)
    # Input vector (r neurons, ternary values)
    inputs = np.array([1, -1, 0, 1])  # Example input vector

    # Layer 1: 3 neurons
    layer1 ={
        "weights": np.array([
            [1, 0, -1, 1],
            [-1, 1, 0, -1],
            [0, 1, 1, -1]
        ]),
        "biases": np.array([0, 1, -1])
    }

    # Layer 2: 2neurons
    layer2 = {
        "weights": np.array([
            [1, -1, 0],  # neuron 1
            [-1, 1, 1]  # neuron 2
        ]),
        "biases": np.array([0, 0])
    }

    # Build network
    net = TernaryNN([layer1, layer2])
    # Run inference
    outputs, trace_data = net.forward(inputs, trace=True)

    print("Final outputs", outputs)

    # Animate propagation
    net.animate(trace_data, interval=1200)