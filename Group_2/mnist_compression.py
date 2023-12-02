from mnist_naive_bayes import NaiveBayes
from mnist_logistic_reg import Logistic_Regression,LogisticRegression
from mnist_extract import *

def test_naive():
    neva=NaiveBayes()
    neva.fit(compression([train_X[0]]),[train_y[0]])
    a=neva.predict(compression([heldout_X[0]]))


    n=len(a)
    accuracy=sum([1 if a[i]==test_y[i]else 0 for i in range(n)])/n
    print(accuracy)

def test_logistic():
    neva=LogisticRegression(lr=0.01)
    neva.fit(compression([train_X[0]]),[train_y[0]])
    a=neva.predict(compression([heldout_X[0]]))


    n=len(a)
    print(a)
    accuracy=sum([1 if a[i]==heldout_y[i]else 0 for i in range(n)])/n
    print(accuracy)


def try_smoothing(smoothings):
    accuracies = []
    for smoothing in smoothings:
        neva=NaiveBayes(smoothing)

        neva.fit(compression(train_X),train_y)
        a=neva.predict(compression(heldout_X))



        n=len(a)
        accuracy=sum([1 if a[i]==heldout_y[i]else 0 for i in range(n)])/n
        accuracies.append(accuracy)
    return accuracies



def try_learningrate(learningrates):
    accuracies = []
    for lr in learningrates:
        neva=LogisticRegression(lr)

        neva.fit(compression(train_X),train_y)
        a=neva.predict(compression(heldout_X))

        
        n=len(a)
        accuracy=sum([1 if a[i]==heldout_y[i]else 0 for i in range(n)])/n
        accuracies.append(accuracy)
    return accuracies


def compression_measurements(smoothings=[1], Lrs=[1]):
    s_accuracies = try_smoothing(smoothings)
    l_accuracies = try_learningrate(Lrs)
    return (s_accuracies, l_accuracies)

# print(compression_measurements())