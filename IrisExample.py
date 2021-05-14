from NN.Layers import PerceptronLayer
from NN.Optimization import gradient_descent, log_loss
from NN.Math import *
import pandas as pd


def test_model(X, Y, model, threshold):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(Y.shape[0]):
        pred = model.forward(X[i])
        prediction = (pred > threshold)
        answer = bool(Y[i])
        if prediction and answer:
            tp += 1
        elif not prediction and not answer:
            tn += 1
        elif not prediction and answer:
            fn += 1
        elif prediction and not answer:
            fp += 1
    return tp, tn, fp, fn


data = pd.read_csv('data/iris.csv', sep=',', names=['0', '1', '2', '3', '4'])
data['4'] = data['4'].map({'Iris-versicolor': 0, 'Iris-setosa': 1})
data = data.sample(frac=1)
X = data[['0', '1', '2', '3']].to_numpy()
Y = data['4'].to_numpy()
model = PerceptronLayer(1, 4, sigmoid, init=uniform_init)
losses = gradient_descent(X, Y, model, 10, 0.002, log_loss)
print(losses)

tp, tn, fp, fn = test_model(X, Y, model, 0.5)
print('True Positive: ', tp, ' True Negative: ', tn, ' False Positive: ', fp, ' False Negative: ', fn)
