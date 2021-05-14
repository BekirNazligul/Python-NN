import numpy as np
from typing import Callable


class Layer:
    def __init__(self, size: int, input_size: int):
        self.size = size
        self.input_size = input_size

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dY: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PerceptronLayer(Layer):
    """
    Constructor function.
    :param size: number of perceptron in this layer.
    :param input_size: the size of the expected feature vector.
    :param activation: The non-linear activation function that
    provides a forward and backward pass option.
    :param init: Optional initializer function for the weights.
    """
    def __init__(self, size: int, input_size: int, activation: Callable, init: Callable = np.random.random_sample):
        super().__init__(size, input_size)
        self.size = size
        self.input_size = input_size
        self.W = init((size, input_size))
        self.b = init((1, size))
        self.activation = activation
        self.layers = [self]
        # Cached gradients for update.
        self.cdW = None
        self.cdb = None
        # Cached forward pass variables for backpropagation.
        self.cX = None
        self.cY = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass function implementing the formula activation(X.W + b).
        Caches the variables for the backwards pass.
        :param X: Feature vector. Must be the same size as W.
        :return: activation(X.W + b)
        """
        X = np.atleast_2d(X)
        self.cX = X
        self.cY = np.dot(X, self.W.T) + self.b
        return self.activation(True, self.cY)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Backwards pass function.
        Caches the variables for the update.
        :param dY: Gradient of the last output.
        :return: The gradient of the inputs.
        """
        dY = np.atleast_2d(dY)
        dact = self.activation(False, self.cY) * dY
        self.cdb = np.sum(dact, axis=0)
        self.cdW = np.dot(dact.T, self.cX)
        dX = np.dot(dact, self.W)
        return dX
