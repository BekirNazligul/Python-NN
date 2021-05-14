import numpy as np


# Activation Functions

def sigmoid(direction: bool, value: np.ndarray) -> np.ndarray:
    """
    :param direction: True for the sigmoid function, False for the derivative.
    :param value:The value operated on. Single value or numpy array.
    Expected to be the last input to this function if false.
    :return: The result of The logistic sigmoid function, or its derivative.
    """
    v = np.clip(value, a_min=-709, a_max=None)  # The received value is clipped to prevent overflow issues.
    if direction:
        return 1 / (1 + np.exp(-v))
    else:
        v = sigmoid(True, v)
        return v * (1 - v)


def ReLU(direction: bool, value: np.ndarray) -> np.ndarray:
    """
    The rectifier linear unit function.
    :param direction: True for the ReLU function, False for the derivative.
    :param value: The value operated on. Single value or numpy array.
    Expected to be the last input to this function if false.
    :return: The result of The ReLU function, or its derivative.
    """
    if direction:
        return np.clip(value, a_min=0, a_max=None)
    else:
        return np.heaviside(value, 1)


# Initialization Functions

def uniform_init(shape):
    return np.random.uniform(-1, 1, shape)


def he_init(shape):
    return np.random.standard_normal(shape) * np.sqrt(2 / shape[1])


def xavier_init(shape):
    return np.random.standard_normal(shape) * np.sqrt(6 / sum(shape))
