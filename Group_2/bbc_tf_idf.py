from bbc_extract import *
from bbc_naive_bayes import Naive_Bayes

import math
import math
class LogisticRegression:
    def __init__(self, learning_rate=1, n_iter=6):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.word_index = None
        self.idf = None

    def fit(self, features, labels):
        unique_labels = list(set(labels))
        all_words = set(word for x in features for word in x.keys())
        self.word_index = {word: i for i, word in enumerate(all_words)}
        self.weights = {label: [0] * len(all_words) for label in unique_labels}
        self.bias = {label: 0 for label in unique_labels}
        
        # Calculate IDF values
        self.idf = {word: self.calculate_idf(word, features) for word in all_words}
        
        for _ in range(self.n_iter):
            for x, y in zip(features, labels):
                predictions = {label: self.sigmoid(self.calculate_score(x, label)) for label in unique_labels}
                errors = {label: predictions[label] - (1 if label == y else 0) for label in unique_labels}
                
                for word in x:
                    if word in self.word_index:
                        j = self.word_index[word]
                        x_word = x[word]
                        idf_word = self.idf[word]
                        for label in unique_labels:
                            dw = errors[label] * x_word * idf_word
                            self.weights[label][j] -= self.learning_rate * dw
                            self.bias[label] -= self.learning_rate * errors[label]

    def calculate_idf(self, word, documents):
        count = sum(1 for document in documents if word in document)
        return math.log(len(documents) / (count + 1))  # Add 1 to avoid division by zero

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
            scores = {label: self.calculate_score(x, label) for label in self.weights.keys()}
            probabilities = {label: self.sigmoid(score) for label, score in scores.items()}
            predicted_label = max(probabilities, key=probabilities.get)
            predictions.append(predicted_label)
        return predictions

    def sigmoid(self, z):# for overflow problem
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        else:
            return math.exp(z) / (1 + math.exp(z))

# from logistic_reg import LogisticRegression

x_train=tf_idf(train_X)
y_train=train_y
x_heldout=tf_idf(heldout_X)
y_heldout=heldout_y
x_test=tf_idf(test_X)
y_test=test_y

def try_smoothing(smoothings):
    accuracies = []

    for smoothing in smoothings:
        neva =Naive_Bayes(smoothing)
        neva.fit(x_train,y_train)
        a=neva.predict(x_heldout)
        n=len(a)
        accuracy=sum([1 if a[i]==y_heldout[i] else 0 for i in range(n)])/n
        accuracies.append(accuracy)

    return accuracies

def test_naive():# we concluded the best smoothing is the 0.001 from try_smoothing function
    neva =Naive_Bayes(0.001)
    neva.fit(x_train+x_heldout,y_train+y_heldout)# in testing we use both heldout and train to train since we chose our smoothing
    a=neva.predict(x_test)
    n=len(a)
    accuracy=sum([1 if a[i]==y_test[i] else 0 for i in range(n)])/n
    # print(accuracy)

def test_log_reg(): # it only works for lr=1
    log_reg=LogisticRegression()
    log_reg.fit(x_train,y_train)
    a=log_reg.predict(x_test)
    n=len(a)
    accuracy=sum([1 if a[i]==y_test[i] else 0 for i in range(n)])/n
    # print(accuracy)

def try_learningrate(Lrs):
    accuracies = []
    for Lr in Lrs:
        Log_reg =LogisticRegression(Lr)
        Log_reg.fit(x_train,y_train)
        a=Log_reg.predict(x_heldout)
        n=len(a)
        # print(a)
        accuracy=sum([1 if a[i]==y_heldout[i] else 0 for i in range(n)])/n
        accuracies.append(accuracy)
    
    return accuracies

def tf_idf_measurements(smoothings=[0.05,0.5,1,10,100], Lrs=[0.0001,0.001,0.01,0.1,1,1.5]):
    s_accuracies = try_smoothing(smoothings)
    l_accuracies = try_learningrate(Lrs)
    return (s_accuracies, l_accuracies)


# print(tf_idf_measurements())
# try_smoothing()
# test_naive()