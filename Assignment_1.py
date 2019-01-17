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

#QUESTION 3 - 0/1.0
#Should work?
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


#QUESTION 4 - 0/1.0
#Should work?
class Queue:
    def __init__(self):
        self.items = []
        
    def push(self,item):
        self.items.insert(0,item)
        
    def pop(self):
        return self.items.pop()
        
    def checkSize(self):
        return len(self.list)

#QUESTION 5 - 0/1.0
#Should work?
import numpy as np

def generateMatrix(r,c,min_val,max_val):
    np.random.seed(0)
    #your code below
    return np.random.randint(low = min_val,high = max_val, size=(r, c))

#QUESTION 6 - .5/1.0
#Half right, assume the error isn't returning correctly?
import numpy as np

def multiplyMat(a, b):
    if(np.size(a, 1) != np.size(b, 0)):
        return("Incompatible Matrices")
    else:
        return a.dot(b)

#print(multiplyMat(generateMatrix(3,3,1,10),generateMatrix(2,2,1,10)))


#QUESTION 7 - 0/1.0
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

    return [sum_a, mean_a, min_a, max_a, sum_b, mean_b, min_b, max_b, pearson, rho]

#print(statsTuple([1, 2, 3, 4], [2, 3, 4, 5]))


#QUESTION 8 - 0/1.0
import pandas as pd
def pandas_func(file_name):
#your code below
    df = pd.read_csv(file_name,sep='\t')

    means = []
    names = []

    for i in df.columns:
        if(df[i].dtype == np.float64 or df[i].dtype == np.int64):
            means.append(df[i].mean())
        else:
            names.append(i)

    return means, names

print(pandas_func('ExampleTab.txt'))

