#QUESTION 1 - 1.0/1.0
def add(a,b):
    if (isinstance(a, (int, float, str, list)) and isinstance(b, (int, float, str, list))):
        if(isinstance(a, (str)) or isinstance(b, (str))):
            return str(a)+str(b)
        return a+b
    else:
        print("Error!")
        return None

#QUESTION 2 - 1.0/1.0
import numpy as np

def calcMyGrade(assign_scores, midterm_scores, practicum_scores, weights):
    assign_grade = np.mean(assign_scores)*weights[0]
    midterm_grade = np.mean(midterm_scores)*weights[1]
    practicum_grade = np.mean(practicum_scores)*weights[2]
    return float(assign_grade + midterm_grade + practicum_grade)/100

#QUESTION 3 - 1.0/1.0
class Node:
    def __init__(self,key,value):
        self.key = key
        self.value = value
        self.leftchild = None
        self.rightchild = None
        self.parent = None        

        
    def getChildren(self):
        return [self.leftchild, self.rightchild]
    
    def getKey(self):
        return self.key


#QUESTION 4 - 1.0/1.0
class Queue:

    def __init__(self):
        self.queue = list()

    def push(self,data):
        self.queue.insert(0,data)

    def pop(self):
        if len(self.queue)>0:
            self.queue.pop()

    def checkSize(self):
        return len(self.queue)

#QUESTION 5 - 1.0/1.0
import numpy as np

def generateMatrix(r,c,min_val,max_val):
    np.random.seed(0)
    #your code below
    return np.random.random_integers(low = min_val-1, high = max_val+1, size=(r, c))

#QUESTION 6 - 1.0/1.0
import numpy as np

def multiplyMat(a, b):
    if(np.size(a, 1) != np.size(b, 0)):
        return("Incompatible Matrices")
    else:
        return a.dot(b)


#QUESTION 7 - 1.0/1.0
import numpy as np
from scipy.stats import pearsonr,spearmanr

def statsTuple(a,b):
    #your code below
    sum_a = sum(a)
    mean_a = np.mean(a)
    min_a = min(a)
    max_a = max(a)
    sum_b = sum(b)
    mean_b = np.mean(b)
    min_b = min(b)
    max_b = max(b)

    pearson, p = pearsonr(a,b)
    rho, pval = spearmanr(a,b)

    return (sum_a, mean_a, min_a, max_a, sum_b, mean_b, min_b, max_b, pearson, rho)


#QUESTION 8 - 1.0/1.0
import pandas as pd
def pandas_func(file_name):
#your code below
    df = pd.read_csv(file_name,sep='\t')

    ListOfMeans = list()
    ListOfColumnNames = list()

    for i in df.columns:
        try:
            ListOfMeans.append(round(pd.to_numeric(df[i]).mean(), 2))
        except:
            ListOfColumnNames.append(i)

    return (ListOfMeans, ListOfColumnNames)


