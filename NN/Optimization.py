import numpy as np
from typing import Callable, List


# Loss functions

def log_loss(direction: bool, prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    The logarithmic loss function.
    :param direction: True for the normal function, False to get the derivative.
    :param prediction: Predicted value for the input. Between 0 and 1.
    :param labels: Correct value for the given input. Expected to be 0 or 1.
    :return: Logarithmic loss for the given parameters if the direction is True.
    Gradient of the prediction value if direction is False.
    """
    labels = labels.reshape(prediction.shape)
    prediction = np.clip(prediction, 1e-10, 1 - 1e-10)
    if direction:
        return sum(-1.0 * (labels * np.log(prediction) + (1 - labels) * (np.log(1 - prediction)))) / labels.size
    else:
        num = prediction - labels
        den = prediction * (1 - prediction)
        return sum(num / den) / labels.size


# Optimization Functions

def gradient_descent(X: np.ndarray, Y: np.ndarray, model, epochs: int, lr: float, loss: Callable,
                     batch_size: int = None) -> List[float]:
    """
    Gradient Descent optimization algorithm.
    :param X: Feature set. Expected to be a 2d numpy array.
    :param Y: Label set.
    :param model: model: The model being optimized.
    :param epochs: epochs: Total number of passes.
    :param lr: Learning rate meta-parameter.
    :param loss: loss: Loss function.
    :param batch_size: Optional batch size parameter. Entire dataset is used if not given.
    :return: A list of average losses per epoch.
    """
    losses = []
    for i in range(epochs):
        losses.append(0)
        if batch_size is None:
            batch_size = Y.size
        perm = np.random.permutation(Y.shape[0])
        xbatch = np.array_split(X[perm], Y.shape[0] / batch_size)
        ybatch = np.array_split(Y[perm], Y.shape[0] / batch_size)
        for x, y in zip(xbatch, ybatch):
            pred = model.forward(x)
            losses[i] += loss(True, pred, y)
            model.backward(loss(False, pred, y))
            for layer in model.layers:
                layer.W -= lr * layer.cdW
                layer.b -= lr * layer.cdb
            losses[i] = losses[i] / (Y.shape[0] / batch_size)
    return losses
