import math
from bbc_extract import *

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=6):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.word_index = None

    def fit(self, features, labels):
        unique_labels = list(set(labels))
        all_words = set(word for x in features for word in x.keys())
        self.word_index = {word: i for i, word in enumerate(all_words)}
        self.weights = {label: [0] * len(all_words) for label in unique_labels}
        self.bias = {label: 0 for label in unique_labels}
        
        for _ in range(self.n_iter):
            for x, y in zip(features, labels):
                predictions = {label: self.sigmoid(self.calculate_score(x, label)) for label in unique_labels}
                errors = {label: predictions[label] - (1 if label == y else 0) for label in unique_labels}
                
                for word in x:
                    if word in self.word_index:
                        j = self.word_index[word]
                        x_word = x[word]
                        for label in unique_labels:
                            dw = errors[label] * x_word
                            self.weights[label][j] -= self.learning_rate * dw
                            self.bias[label] -= self.learning_rate * errors[label]


    def calculate_score(self, x, y):
        score = self.bias[y]
        for word in x:
            if word in self.word_index:
                j = self.word_index[word]
                score += self.weights[y][j] * x[word]
        return score

    def predict(self, features):
        predictions = []
        for x in features:
            scores = {}
            for label in self.weights.keys():
                score = self.calculate_score(x, label)
                scores[label] = self.sigmoid(score)
            predicted_label = max(scores, key=scores.get)
            predictions.append(predicted_label)
        return predictions

    def sigmoid(self, z):
        return 0.5 * (1 + math.tanh(z / 2))
