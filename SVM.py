

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random as rd
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split


max_iterations = 40
#C is the upperbound of the lagrange multiplier
C = 4

#linear case
first_cluster_x1_mean=10
first_cluster_x1_dev=1
first_cluster_y1_mean=10
first_cluster_y1_dev=1

first_cluster_x2_mean=16
first_cluster_x2_dev=1
first_cluster_y2_mean=16
first_cluster_y2_dev=1

#non linear case
noise=0.05
factor=0.4

#bound of the error
error_bound=10**-3
lag_bound=10**-5




class SVM:
    
    def __init__(self,data,case,n_points=500):
        self.case=case
        self.data=data
        if (self.data == "linear_iris"):
            self.X, self.y, self.X_test, self.y_test = self.generate_linear_data_iris()
            self.lagranges = np.zeros(self.X.shape[0])
            self.b = 0
            self.w=np.array([0,0])
        elif (self.data == "non_linear_iris"):
            self.X, self.y, self.X_test, self.y_test = self.generate_non_linear_data_iris()
            self.lagranges = np.zeros(self.X.shape[0])
            self.b = 0
            self.w=np.array([0,0])
        elif (self.data=="linear"):
            self.X, self.y = self.generate_linear_data(n_points)
            self.lagranges = np.zeros(self.X.shape[0])
            self.b = 0
            self.w=np.array([0,0])
        elif (self.data=="non_linear"):
            self.X, self.y = self.generate_non_linear_data(n_points)
            self.lagranges = np.zeros(self.X.shape[0])
            self.b = 0
            self.w=np.array([0,0])
        
        
    def generate_non_linear_data(self,n_points):
        X,y = datasets.make_circles(n_samples=n_points, shuffle=True, noise=noise, factor=factor)
        y[y == 0]=-1
        return X,y
        
    def generate_linear_data(self,n_points):
        clusters = []
        for i in range(int(n_points/2)):
            clusters.append([rd.gauss(first_cluster_x1_mean,first_cluster_x1_dev),
                             rd.gauss(first_cluster_y1_mean,first_cluster_y1_dev), 1])
        for i in range(int(n_points/2)):
             clusters.append([rd.gauss(first_cluster_x2_mean,first_cluster_x2_dev),
                             rd.gauss(first_cluster_y2_mean,first_cluster_y2_dev), -1])
        data = pd.DataFrame(np.array(clusters),columns=['x1', 'x2', 'y'])
        
        X = np.array(data.drop('y',axis=1))
        y = np.array(data['y'])
        return X,y
    
    def generate_linear_data_iris(self):
        iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = pd.read_csv(iris_url, sep = ',', header = None\
                   , names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species'])
        iris = iris[ iris.species != 'Iris-virginica']
        X = iris.drop(['species','sepal width','sepal length' ], axis=1)
        y = iris['species']
        
        mapper = {
            "Iris-setosa" : -1,
            "Iris-versicolor" : 1
            }
        
        y = iris['species'].map(mapper)
        X, X_test, y, y_test = train_test_split(X, y, train_size = 0.75)
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        return X,y,X_test,y_test
    
    
    def generate_non_linear_data_iris(self):
        iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        iris = pd.read_csv(iris_url, sep = ',', header = None\
                   , names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species'])
        iris = iris[ iris.species != 'Iris-setosa']
        X = iris.drop(['species','sepal width','sepal length' ], axis=1)
        y = iris['species']
        
        mapper = {
            "Iris-virginica" : -1,
            "Iris-versicolor" : 1
            }
        
        y = iris['species'].map(mapper)
        X, X_test, y, y_test = train_test_split(X, y, train_size = 0.75)
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        return X,y,X_test,y_test
    
    
    
    def smo(self):
        i=0
        while (i < max_iterations):
            if (self.do_smo()==0):
                i+=1
            else:
                i=0
        if (self.case=="linear"):
            self.w = self.find_w(self.lagranges,self.X,self.y)
            
        
    def do_smo(self):
        operations=0
        for i in range(self.X.shape[0]):
            Ei = np.dot(self.y*self.lagranges,(self.kernel_function(self.X[i],self.X))) + self.b - self.y[i]  
            #go to slide 1
            if (self.lagranges[i] < C and self.y[i]*Ei < (-error_bound ) or \
                (self.lagranges[i] > 0 and self.y[i]*Ei > (error_bound))):
                 
               j = self.chooseIndexj(i, len(self.lagranges))
               Ej = np.dot(self.y*self.lagranges,(self.kernel_function(self.X[j],self.X))) + self.b - self.y[j]     
               oldLag_i = self.lagranges[i]
               oldLag_j = self.lagranges[j]
               #go to slide 2
               lowerBound, higherBound = self.find_bounds(self.y[i], self.y[j], self.lagranges[i], 
                                                            self.lagranges[j])
               
               if (lowerBound != higherBound):
                   eta = 2 * self.kernel_function(self.X[i],self.X[j]) - self.kernel_function(self.X[i],self.X[i]) \
                   - self.kernel_function(self.X[j],self.X[j]) 
                   if eta < 0:
                           self.lagranges[j] = self.optimize_lagj(self.lagranges[j],self.y[j],
                                                                      Ei,Ej,eta,higherBound,lowerBound)
                           if (np.abs(self.lagranges[j] - oldLag_j) > lag_bound):
                                self.lagranges[i] = self.optimize_lagi(self.lagranges[i],self.lagranges[j],
                                                                           oldLag_j,self.y[i],self.y[j])
                                self.b = self.find_b(oldLag_i,oldLag_j,Ei,Ej,i,j)
                                operations +=1
        return operations
    
   
    def find_b(self,oldLag_i,oldLag_j,Ei,Ej,i,j):
        b1 = self.b - Ei - self.y[i] * (self.lagranges[i] - oldLag_i) * self.kernel_function(self.X[i],self.X[i]) - \
                           self.y[j] * (self.lagranges[j] - oldLag_j) * self.kernel_function(self.X[i],self.X[j])
        b2 = self.b - Ej - self.y[i] * (self.lagranges[i] - oldLag_i) * self.kernel_function(self.X[i],self.X[j]) - \
                           self.y[j] * (self.lagranges[j] - oldLag_j) * self.kernel_function(self.X[j],self.X[j])
        if (0< self.lagranges[i]) and (self.lagranges[i] < C):
            ris=b1
        elif (0<self.lagranges[j]) and (self.lagranges[j]<C):
            ris=b2
        else:
            ris=(b1+b2)/2
        return ris 
   

    def optimize_lagj(self,lagj,yj,Ei,Ej,eta,higherBound,lowerBound):
        lagj = lagj - (yj*(Ei-Ej))/eta
        if lagj > higherBound:
            lagj=higherBound
        elif lagj < lowerBound:
            lagj=lowerBound
        return lagj
    
    def optimize_lagi(self,lag_i,lag_j,oldLag_j,yi,yj):
        lag_i  = lag_i + yi*yj*(oldLag_j - lag_j)
        return lag_i

    def chooseIndexj(self,indexLagrage_i,size):
        possible = list(range(size))
        possible.pop(indexLagrage_i)
        return rd.choice(possible)
    
    def find_bounds(self,yi,yj,lag_i,lag_j):
        if (yi==yj):
            lowerBound = max(0,lag_j+lag_i-C)
            higherBound = min(C,lag_i+lag_j)
        else:
            lowerBound = max(0,lag_j-lag_i)
            higherBound = min(C,C+lag_j-lag_i)
        return lowerBound, higherBound
    
    def find_w(self,lagrange,X,y):
        w=np.array([0,0])
        for i in range(len(y)):
            w = w + lagrange[i]*y[i]*X[i]
        return w
    
    def kernel_function(self,x,y):
        if self.case=="linear":
            return np.dot(x,np.transpose(y))
        if self.case=="non_linear":
            return (np.dot(x,np.transpose(y)) ** 2)
        
        
        
    def plot_data(self):
        colors=[]
        error=0
        for i in range(len(self.X)):
            ris = svm.predict((self.X)[i])
            if ris!= self.y[i]:
                colors.append("red")
                error+=1
            elif ris==1:
                colors.append("green")
            elif ris==-1:
                colors.append("magenta")
        red_patch = mpatches.Patch(color='red', label='Data misclassified')
        green_patch = mpatches.Patch(color='green', label='Y = 1')
        magenta_patch = mpatches.Patch(color='magenta', label='Y = -1')
        
        if ((self.data == "linear_iris" or self.data== "non_linear_iris" or \
             self.data == "linear" or self.data == "non_linear") and self.case=="linear"):
            if (self.data == "linear_iris"):
                x = np.linspace(1,5,100)
            elif (self.data == "non_linear_iris"):
                x = np.linspace(3,7,100)
            elif (self.data == "linear"):
                x = np.linspace(7,20,100)
            elif (self.data == "non_linear"):
                x = np.linspace(-1.3,1.3,100)
            fig = plt.figure()
            plt.plot(x,((-self.w[0] * x - self.b )/self.w[1]) , color="blue")
            plt.scatter(self.X[:,0],self.X [:,1],c=colors)
            plt.legend(loc="upper left",handles=[red_patch,green_patch,magenta_patch])
            plt.grid()
            plt.show()
        elif ((self.case == "non_linear_iris" or self.case=="non_linear" or \
               self.case=="linear_iris" or self.case=="linear") and self.case=="non_linear"):
            fig = plt.figure()
            plt.scatter(self.X[:,0],self.X [:,1],c=colors)
            plt.legend(loc="upper left",handles=[red_patch,green_patch,magenta_patch])
            plt.grid()
            plt.show()
        print("Size of training set: "+str(len(self.X)))
        print("Incorrect predictions: "+str(error))
            
       
    def predict(self,x):
        return (np.sign(np.dot(self.y*self.lagranges,(self.kernel_function(x,self.X))) + self.b))
       
    def test(self):
        error=0
        if self.data=="linear":
            self.X_test, self.y_test = self.generate_linear_data(126)
        elif self.data=="non_linear" :
            self.X_test, self.y_test = self.generate_non_linear_data(126)
                
        for i in range(len(self.X_test)):
            ris = svm.predict((self.X_test)[i])
            if ris != (self.y_test)[i]:
                error+=1
        print("Size of test set: "+str(len(self.X_test)))
        print("Incorrect predictions: "+str(error))
    
            
       

svm = SVM("linear","linear")
#svm = SVM("non_linear","linear")
#svm = SVM("non_linear","non_linear")
#svm = SVM("linear_iris","linear")
#svm = SVM("non_linear_iris","linear")
#svm = SVM("non_linear_iris","non_linear")
svm.smo()
svm.plot_data()
svm.test()


