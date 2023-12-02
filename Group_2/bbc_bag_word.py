from bbc_extract import *
from bbc_naive_bayes import Naive_Bayes
from bbc_logistic_reg import LogisticRegression

neva=Naive_Bayes(0.1)
neva.fit(train_X,train_y)

def try_smoothing(smoothings):
    accuracies = []
    for smoothing in smoothings:
        neva =Naive_Bayes(smoothing)
        neva.fit(train_X,train_y)
        a=neva.predict(heldout_X)
        n=len(a)
        accuracy=sum([1 if a[i]==heldout_y[i] else 0 for i in range(n)])/n
        accuracies.append(accuracy)
    
    return accuracies

def testnaive():# we concluded the best smoothing is the 0.1
    neva =Naive_Bayes(0.1)
    neva.fit(train_X+heldout_X,train_y+heldout_y)
    a=neva.predict(test_X)
    n=len(a)
    accuracy=sum([1 if a[i]==test_y[i] else 0 for i in range(n)])/n
    return accuracy
    # print(accuracy)

def test_log_reg():
    log_reg=LogisticRegression()
    log_reg.fit(train_X,train_y)
    a=log_reg.predict(test_X)
    n=len(a)
    accuracy=sum([1 if a[i]==test_y[i] else 0 for i in range(n)])/n
    # print(accuracy)

def try_learningrate(Lrs):
    accuracies = []

    for Lr in Lrs:
        Log_reg =LogisticRegression(Lr)
        Log_reg.fit(train_X,train_y)
        a=Log_reg.predict(heldout_X)
        n=len(a)
        accuracy=sum([1 if a[i]==heldout_y[i] else 0 for i in range(n)])/n
        accuracies.append(accuracy)
    
    return accuracies
        


def bow_measurements(smoothings=[0.05,0.5,1,10,100], Lrs=[0.0001,0.001,0.01,0.1,1,1.5]):
    s_accuracies = try_smoothing(smoothings)
    l_accuracies = try_learningrate(Lrs)
    return (s_accuracies, l_accuracies)


# print(bow_measurements())

# try_learningrate()
# try_smoothing()
# test()