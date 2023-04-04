# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import all the libraries which are needed to the program.
    get profit prediction graph and computecost value.
    Get a graph of cost function using gradient descent and also get profit prediction graph.
    Get the otput of profit for the population of 35,000 and 70,000.

 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: v.sreeja
RegisterNumber:  212222230169
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Predication")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
  data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history
  
  theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000s")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]
    
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))


predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

```

## Output:
![Screenshot (97)](https://user-images.githubusercontent.com/118344328/229812805-bf40dc37-dc4f-4a76-84b0-51da2a82f183.png)
![Screenshot (98)](https://user-images.githubusercontent.com/118344328/229812969-a51380ed-1d67-4d4d-9ab3-5c346c67bd90.png)
![Screenshot (99)](https://user-images.githubusercontent.com/118344328/229813109-fb8d981e-a1e6-4eb3-98ab-d433e0e3af00.png)
![Screenshot (100)](https://user-images.githubusercontent.com/118344328/229813490-974043fd-70f0-4ac1-9848-b1b1669b5f46.png)
![Screenshot (101)](https://user-images.githubusercontent.com/118344328/229813552-9daca927-8fc9-4915-9b08-11d760e23641.png)
![Screenshot (102)](https://user-images.githubusercontent.com/118344328/229813661-3b694d7c-7bd7-4b45-a64f-db23b453e9da.png)
![Screenshot (102)](https://user-images.githubusercontent.com/118344328/229813857-161bda7e-68f7-4b0b-9e2d-3a45e650270d.png)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
