from bbc_extract import *
from bbc_naive_bayes import Naive_Bayes
from bbc_logistic_reg import LogisticRegression

x_train=one_hot_encoding(train_X)
y_train=train_y
x_heldout=one_hot_encoding(heldout_X)
y_heldout=heldout_y
x_test=one_hot_encoding(test_X)
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

def testbayes():# we concluded the best smoothing is the 0.07
    neva =Naive_Bayes(0.07)
    neva.fit(x_train+x_heldout,y_train+y_heldout)
    a=neva.predict(x_test)
    n=len(a)
    accuracy=sum([1 if a[i]==y_test[i] else 0 for i in range(n)])/n
    # print(accuracy)

def test_log_reg():
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
        accuracy=sum([1 if a[i]==y_heldout[i] else 0 for i in range(n)])/n
        accuracies.append(accuracy)

    return accuracies

def one_hot_measurements(smoothings=[0.05,0.5,1,10,100], Lrs=[0.0001,0.001,0.01,0.1,1,1.5]):
    s_accuracies = try_smoothing(smoothings)
    l_accuracies = try_learningrate(Lrs)
    return (s_accuracies, l_accuracies)

    
# print(one_hot_measurements())

