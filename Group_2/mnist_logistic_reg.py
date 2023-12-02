import math
class Logistic_Regression:# one vs all approach
    def __init__(self,lr=0.01,n_iter=2):
        self.lr=lr
        self.n_iter=n_iter
        self.weights={} #dictionary of array
        self.bias={}# dictionary

    def predict(self,X):
        predictions=[]
        for x in X:
            x_prediction={}
            for label in self.weights:
                label_prediction=self._predict(label,x)
                x_prediction[label]=label_prediction
            prediction = max(x_prediction, key=x_prediction.get)
            predictions.append(prediction)
        return predictions
    
    def _predict(self,label,x):
        Z=sum([self.weights[label][i]*x[i] for i in range(len(x))])+self.bias[label]
        return 1 / (1 + math.exp(-Z))
    
    def _transform(self,y):
        transform_vector=[]
        for label in self.weights:
            if label==y:
                transform_vector.append(1)
            else:
                transform_vector.append(0)
        return transform_vector
    
    def fit(self,X,Y):
        n_features=len(X[0])
        unique_labels=set(Y)
        for label in unique_labels:
            self.bias[label]=0
            self.weights[label]=[0]*n_features
        y_transformed=[self._transform(y) for y in Y]# this changes y from a label to [0,1,0...] so we can work on each regression
        y_classified=list(zip(*y_transformed))
        label_values = {key: value for key, value in zip(self.weights.keys(),y_classified)}
        for label in label_values:
            self._fit(label,X,label_values[label])
    
    def _fit(self,label,X,Y):
        n_features=len(X[0])
        n=len(Y)
        for _ in range(self.n_iter):
            predicted_values=[self._predict(label,x) for x in X]
            for j in range(n_features):
                dw=0.0
                for i in range(n):
                    dw += (predicted_values[i] - Y[i]) * X[i][j]
                dw /= n
                self.weights[label][j] -= self.lr * dw
            db = 0.0
            for i in range(n):
                db += predicted_values[i] - Y[i]
            db /= n
            self.bias[label] -= self.lr * db
import math

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1): #Took too long for more n_iter
        self.lr = lr
        self.n_iter = n_iter
        self.weights={}
        self.bias = {}
        self.labels=[i for i in range(10)]
        for label in self.labels:
            self.weights[label]=[]
            self.bias[label]=0

    def predict(self, X):
        predictions = []
        for x in X:
            x_prediction = {}
            for label in self.labels:
                label_prediction = self._predict(label, x)
                x_prediction[label] = label_prediction
            prediction = max(x_prediction, key=x_prediction.get)
            predictions.append(prediction)
        return predictions
    def _transform(self,y):
        transform_vector=[]
        for label in self.labels:
            if label==y:
                transform_vector.append(1)
            else:
                transform_vector.append(0)
        return transform_vector
    def _predict(self, label, x):
        z = sum([self.weights[label][i] * x[i] for i in range(len(x))]) + self.bias[label]
        return self.sigmoid(z)

    def fit(self, X, Y):
        num_features=len(X[0])
        for label in self.labels:
            self.weights[label]=[0]*num_features
        y_transformed=[self._transform(y) for y in Y]
        y_classified=list(zip(*y_transformed))
        for label in self.labels:
            self._fit(X,y_classified[label],label)

    def _fit(self, X, Y_label, label):
        n = len(Y_label)
        n_features=len(X[0])
        for _ in range(self.n_iter):
            for i in range(n):
                predicted_Y=self._predict(label,X[i])
                error=Y_label[i]-predicted_Y
                for j in range(n_features):
                    self.weights[label][j] += self.lr * error * X[i][j]
                self.bias[label]+=self.lr*error


    def sigmoid(self, z):# for overflow problem
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        else:
            return math.exp(z) / (1 + math.exp(z))    
