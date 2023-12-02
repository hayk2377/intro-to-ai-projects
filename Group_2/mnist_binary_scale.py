from mnist_naive_bayes import NaiveBayes
from mnist_logistic_reg import Logistic_Regression,LogisticRegression
from mnist_extract import *

def test_naive():
    neva=NaiveBayes(smoothing=1)
    neva.fit(binary_scale(train_X),train_y)
    a=neva.predict(binary_scale(test_X))
    n=len(a)
    accuracy=sum([1 if a[i]==test_y[i]else 0 for i in range(n)])/n
    print(accuracy)

def try_smoothing(smoothings):
    accuracies = []

    for smoothing in smoothings:
        neva=NaiveBayes(smoothing)

        neva.fit(binary_scale(train_X),train_y)
        a=neva.predict(binary_scale(heldout_X))


        n=len(a)
        accuracy=sum([1 if a[i]==heldout_y[i]else 0 for i in range(n)])/n
        accuracies.append(accuracy)

    return accuracies


def try_learningrate(learningrates):

    accuracies = []
    for lr in learningrates:
        neva=LogisticRegression(lr)

        neva.fit(binary_scale(train_X),train_y)
        a=neva.predict(binary_scale(heldout_X))

        n=len(a)
        # print(a)
        accuracy=sum([1 if a[i]==heldout_y[i]else 0 for i in range(n)])/n
        accuracies.append(accuracy)

    return accuracies

def binary_scale_measurements(smoothings=[1], Lrs=[1]):
    s_accuracies = try_smoothing(smoothings)
    l_accuracies = try_learningrate(Lrs)
    return (s_accuracies, l_accuracies)

# print(binary_scale_measurements())