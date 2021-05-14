from .Layers import *
from typing import List


class MultiLayerPerceptron:

    def __init__(self, layers: List[Layer]):
        """
        Constructor function.
        :param layers: A list of layers in the network. Each layer must implement a forward and backward function.
        """
        self.layers = layers
        self.check_dims()

    def check_dims(self) -> None:
        """Checks the network dimensions to see if they match. Raises an error if not."""
        for i in range(0, len(self.layers)-1):
            if self.layers[i].size != self.layers[i+1].input_size:
                raise RuntimeError("Network dimensions don't between layers {} - {} ".format(i, i+1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagates through the network.
        :param X: The features for the input layer.
        :return: The result of the final layer.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dY: np.ndarray) -> None:
        """
        Back-propagates through each layer.
        :param dY: The gradient of the loss function.
        """
        for layer in reversed(self.layers):
            dY = layer.backward(dY)
