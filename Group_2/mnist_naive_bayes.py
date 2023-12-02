import math
class NaiveBayes:
    def __init__(self, smoothing=0.001):
        self.ls = smoothing
        self.prior_probability = {}
        self.feature_probability = {}

    def fit(self, featuresList, labels):
        n = len(labels)
        n_feature = len(featuresList[0])
        unique_labels = set(labels)

        for label in unique_labels:
            self.prior_probability[label] = (labels.count(label) + self.ls) / (n + len(unique_labels) * self.ls)
            self.feature_probability[label] = [{} for _ in range(n_feature)]

        for label in self.prior_probability:
            label_features = [featuresList[i] for i in range(n) if labels[i] == label]
            L = len(label_features)

            for features in label_features:
                for i in range(n_feature):
                    self.feature_probability[label][i][features[i]] = self.feature_probability[label][i].get(features[i], 0) + 1

            for i in range(n_feature):
                for feature in self.feature_probability[label][i]:
                    self.feature_probability[label][i][feature] = (self.feature_probability[label][i][feature] + self.ls) / (L + len(self.feature_probability[label][i]) * self.ls)

    def predict(self, features_list):
        predictions = [self._predict(features) for features in features_list]
        return predictions

    def _predict(self, features):
        n = len(features)
        prediction = {}

        for label in self.prior_probability:
            label_probability = math.log(self.prior_probability[label])
            for i in range(n):
                feature_prob = self.feature_probability[label][i].get(features[i], self.ls)
                label_probability += math.log(feature_prob)

            prediction[label] = label_probability

        max_log_prob = max(prediction.values())
        predicted_label = [label for label, log_prob in prediction.items() if log_prob == max_log_prob][0]
        return predicted_label

