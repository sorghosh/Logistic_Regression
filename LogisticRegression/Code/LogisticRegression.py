import numpy as np
import pandas as pd
import scipy.optimize as opt

class logistic_regression :
    def __init__(self,data):
        self.data   = data
        self.data.insert(0,"Ones",1)
        self.shape  = data.shape[1]
        
    def sigmoid(self,z):
        ## this function converts the predictions into probablity
        return 1/(1 + np.exp(-z))
        
    def cost(self,theta,x,y,):
        x      = np.matrix(x)
        y      = np.matrix(y)
        theta  = np.matrix(theta)
        first  = np.multiply(-y, np.log(self.sigmoid(x * theta.T)))
        second = np.multiply((1-y),np.log(1- self.sigmoid(x* theta.T)))
        return np.sum(first - second)/(len(x))
        
    def gradient(self,theta,x,y):
        x          = np.matrix(x)
        y          = np.matrix(y)
        theta      = np.matrix(theta)
        parameters = int(theta.shape[1])
        grad       = np.zeros(self.shape-1)
        error      = self.sigmoid(x * theta.T) - y
        for i in range(parameters):
            term = np.multiply(error , x[:,i])
            grad[i] = np.sum(term)/(len(x))
        return grad
    
    def optimize(self,x,y):
#        x      = np.array(self.data.iloc[:,:self.shape-1])
#        y      = np.array(self.data.iloc[:,self.shape-1:]) 
        theta  = np.zeros(self.shape-1)   
        result = opt.fmin_tnc(func = self.cost, x0 = theta, fprime = self.gradient, args = (x,y))

        return result , self.cost(result[0],x,y)
        
    def predict(self):
        x = np.array(data.iloc[:,0:self.shape-1])
        y = np.array(data.iloc[:,self.shape-1: self.shape])
        result ,cost = self.optimize(x,y)
        predict = self.sigmoid(np.matrix(x) * np.matrix(result[0]).T)
        return pd.DataFrame(predict)


data = pd.read_csv(r"C:\Users\sauravghosh\Desktop\Machine_Learning\Algo\LogisticRegression\Data\data.txt",header = None)
data.columns = ['Exam 1', 'Exam 2', 'Admitted']
obj = logistic_regression(data)
predict = obj.predict()






