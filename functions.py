import numpy as np
import pandas as pd


def train_model(data, weights, bias, l_rate, epochs):
    epoch_loss = []
    bias, l_rate, epochs = float(bias), float(l_rate), int(epochs)
    for e in range(epochs):
        individual_loss = []
        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(feature, weights, bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy(target, prediction)
            individual_loss.append(loss)
            # gradient descent
            weights = update_weights(weights, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)
        average_loss = sum(individual_loss) / len(individual_loss)
        epoch_loss.append(average_loss)
    return epoch_loss


def generate_data(n_features, n_values):
    n_features, n_values = int(n_features), int(n_values)

    rg = np.random.default_rng()
    features = rg.random((n_features, n_values))
    weights = rg.random((1, n_values))[0]
    targets = np.random.choice([0, 1], n_features)
    data = pd.DataFrame(features, columns=[feature for feature in features])
    data["targets"] = targets
    return data, weights


def get_weighted_sum(feature, weights, bias):
    return np.dot(feature, weights) + bias


def sigmoid(w_sum):
    return 1 / (1 + np.exp(-w_sum))


def cross_entropy(target, prediction):
    return -(target * np.log10(prediction) + (1 - target) * np.log10(1 - prediction))


def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x, w in zip(feature, weights):
        new_w = w + l_rate * (target - prediction) * x
        new_weights.append(new_w)
    return new_weights


def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate * (target - prediction)
