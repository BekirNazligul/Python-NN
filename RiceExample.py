from NN.Layers import PerceptronLayer
from NN.Optimization import gradient_descent, log_loss
from NN.Math import *
from NN.Models import MultiLayerPerceptron
import pandas as pd
from sklearn import preprocessing


def preprocess(df):
    df['CLASS'] = df['CLASS'].map({'Cammeo': 0, 'Osmancik': 1})
    df = df.sample(frac=1.0)
    df = df.infer_objects()
    X = df[['AREA', 'PERIMETER', 'MAJORAXIS', 'MINORAXIS', 'ECCENTRICITY', 'CONVEX_AREA', 'EXTENT']].to_numpy()
    Y = df['CLASS'].to_numpy()
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(X), Y


def test_model(X, Y, model, threshold=0.5):
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


traindata = pd.read_csv('ricetrain.csv', sep=',')
testdata = pd.read_csv('ricetest.csv', sep=',')
xtrain, ytrain = preprocess(traindata)
xtest, ytest = preprocess(testdata)
model = MultiLayerPerceptron([PerceptronLayer(15, 7, ReLU, init=xavier_init),
                              PerceptronLayer(20, 15, ReLU, init=xavier_init),
                              PerceptronLayer(1, 20, sigmoid, init=xavier_init)])
losses = gradient_descent(xtrain, ytrain, model, 50, 0.01, log_loss, batch_size=10)
tp, tn, fp, fn = test_model(xtest, ytest, model)
print("True positive: ", tp, "True negative: ", tn, "False positive: ", fp, "False negative: ", fn)
print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn), " Precision: ", tp / (tp + fp), " Recall: ", tp / (tp + fn))
