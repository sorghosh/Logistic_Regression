#load the required libaries 
import numpy  as np 
import pandas as pd
import scipy.optimize as opt


class logistic_regression:
    def __init__(self,data,learning_rate):
        self.data          = data
        self.learning_rate = learning_rate
        self.data.insert(0,"intercpt",1)
        self.shape         = self.data.shape[1]
        self.x = np.array(self.data.iloc[:,0:self.shape -1])
        self.y = np.array(self.data.iloc[:,self.shape -1:])
        
    
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))    
        
    def cost(self,theta, x, y):
        x       = np.matrix(x)
        y       = np.matrix(y)        
        theta   = np.matrix(theta)
        first   = np.multiply(-y,np.log(self.sigmoid(x * theta.T)))
        second  = np.multiply((1-y), np.log(1 - self.sigmoid(x * theta.T)))
        reg     = (self.learning_rate/2 * len(x)) * np.sum((np.power(theta[:,1:theta.shape[1]],2)))
        costs    = np.sum((first - second)/len(x)) + reg
        return costs
        
    def gradient(self,theta, x, y):
        x       = np.matrix(x)
        y       = np.matrix(y)        
        theta   = np.matrix(theta)
        grad    = np.zeros(x.shape[1])
        param   = int(x.shape[1])
        error   = self.sigmoid(x * theta.T) - y
        for i in range(param):
            term = np.multiply(error , x[:,i])
            if i == 0:
                grad[i] = np.sum(term)/len(x)
            else:
                grad[i] = np.sum(term)/len(x) + (self.learning_rate/len(x) * theta[:,i])
        return grad
                                

    def optimize(self,x,y):
        theta  = np.zeros(x.shape[1])
        result =  opt.fmin_tnc(func = self.cost,x0 = theta , fprime = self.gradient , args = (x,y))
        cost   =  self.cost(theta, x,y)
        return result,cost
    
   
    def predict(self):
        result, cost = self.optimize(self.x,self.y)
        prediction = self.sigmoid(self.x * np.matrix(result[0]).T)
        return result, cost , prediction
        
    def accuracy(self,prediction):
        pred = []
        for p in prediction:
            if p >= .5:
                pred.append(1)
            else:
                pred.append(0)
        
        df = pd.DataFrame(pred)
        df.columns = ["Predicted"]
        df["Actuals"] = self.y
        df["Compare"] = df["Predicted"] == df["Actuals"]
        return float(df.Compare.sum())/float(len(df))
        
    
    
data = pd.read_csv(r"C:\Users\sauravghosh\Desktop\Machine_Learning\Algo\LogisticRegression\Data\data.txt", header = None)
data.columns = ['Exam 1', 'Exam 2', 'Admitted']
obj = logistic_regression(data,1)
result, cost, prediction = obj.predict()
accuracy_score = obj.accuracy(prediction)
print accuracy_score


