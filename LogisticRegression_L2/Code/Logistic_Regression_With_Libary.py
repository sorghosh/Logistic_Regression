import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


data = pd.read_csv(r"C:\Users\sauravghosh\Desktop\Machine_Learning\Algo\LogisticRegression\Data\data.txt", header = None)
data.columns = ['Exam 1', 'Exam 2', 'Admitted']

y = data.pop("Admitted")
x = data

log = LogisticRegression(penalty = "l2", solver  = "newton-cg")
log.fit(x,y)
pred = log.predict(x)
print metrics.accuracy_score(y,pred)
##accuracy score is 0.89

