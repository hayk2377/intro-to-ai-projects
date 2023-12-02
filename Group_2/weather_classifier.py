import math
from weather_setup import *


class NaiveBayes:
    def __init__(self, smoothing=0.0001):
        self.smoothing = smoothing
        self.prior_probability = {}
        self.feature_probability = {}

    def train(self, X, y):
        n = len(y)
        n_feature = len(X[0])
        unique_labels = set(y)

        for label in unique_labels:
            self.prior_probability[label] = (
                y.count(label) + self.smoothing) / (n + len(unique_labels) * self.smoothing)
            self.feature_probability[label] = {}

        for label in self.prior_probability:
            label_features = [X[i] for i in range(n) if y[i] == label]
            total_sample = len(label_features)

            for features in label_features:
                for i in range(n_feature):
                    self.feature_probability[label][features[i]] = self.feature_probability[label].get(
                        features[i], 0) + 1

            for feature in self.feature_probability[label]:
                self.feature_probability[label][feature] = (
                    self.feature_probability[label][feature] + self.smoothing) / total_sample

    def predict(self, features_list):
        predictions = [self._predict(features) for features in features_list]
        return predictions

    def _predict(self, features):
        n = len(features)
        prediction = {}

        for label in self.prior_probability:
            label_probability = math.log(self.prior_probability[label])
            for i in range(n):
                feature_prob = self.feature_probability[label].get(
                    features[i], 0)+self.smoothing
                label_probability += math.log(feature_prob)

            prediction[label] = label_probability

        max_log_prob = max(prediction.values())
        predicted_label = [
            label for label, log_prob in prediction.items() if log_prob == max_log_prob][0]
        return predicted_label


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.classifiers = {}

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def dot_product(self, X, weights):
        result = 0
        for i in range(len(X)):
            result += X[i] * weights[i]
        return result

    def train(self, X, y):
        classes = set(y)

        for cls in classes:
            binary_y = [1 if label == cls else 0 for label in y]
            classifier = self._train_binary_classifier(X, binary_y)
            self.classifiers[cls] = classifier

    def _train_binary_classifier(self, X, y):
        num_samples, num_features = len(X), len(X[0])

        weights = [0] * num_features
        bias = 0

        for _ in range(self.num_iterations):
            for i in range(num_samples):
                linear_model = self.dot_product(X[i], weights) + bias
                predicted_prob = self.sigmoid(linear_model)

                for j in range(num_features):
                    weights[j] -= self.learning_rate * \
                        (predicted_prob - y[i]) * X[i][j]
                bias -= self.learning_rate * (predicted_prob - y[i])

        return {"weights": weights, "bias": bias}

    def predict(self, X):
        predictions = []
        for sample in X:
            class_scores = {}
            for cls, classifier in self.classifiers.items():
                linear_model = self.dot_product(
                    sample, classifier["weights"]) + classifier["bias"]
                predicted_prob = self.sigmoid(linear_model)
                class_scores[cls] = predicted_prob

            predicted_label = max(class_scores, key=class_scores.get)
            predictions.append(predicted_label)

        return predictions


def test_classifier(classifier_name):
    classifier = classifier_name

    name = "naive" if isinstance(classifier, NaiveBayes) else "logistic"

    X_train, y_train = parse_data(load_data('train.csv'))
    X_test, y_test = parse_data(load_data('test.csv'))

    encoded_X_train, encoded_X_test = label_encoding(
        X_train, name), label_encoding(X_test, name)
    encoded_y_train, encoded_y_test = encode_labels(
        y_train), encode_labels(y_test)

    classifier.train(encoded_X_train, encoded_y_train)

    predictions = classifier.predict(encoded_X_test)
    wrong_predictions = []

    for i in range(len(predictions)):
        if encoded_y_test[i] != predictions[i]:
            wrong_predictions.append(
                f"{predictions[i]} was expected to be {encoded_y_test[i]}")

    accuracy = 100 - ((len(wrong_predictions) / len(predictions)) * 100)
    # print(f"Accuracy: {accuracy}%")

    return accuracy





def weather_measurements(smoothings=[0.05,0.5,1,10,100], learning_rates=[0.0001,0.001,0.01,0.1,1,1.5]):
    s_accuracies = []
    l_accuracies = []

    for smoothing in smoothings:
        accuracy = test_classifier(NaiveBayes(smoothing))
        s_accuracies.append(accuracy)
    
    for learning_rate in learning_rates:
        accuracy = test_classifier(LogisticRegression(learning_rate))
        l_accuracies.append(accuracy)
    
    return (s_accuracies, l_accuracies)