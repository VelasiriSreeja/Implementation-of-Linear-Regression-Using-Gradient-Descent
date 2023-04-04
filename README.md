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
https://user-images.githubusercontent.com/119288183/229331834-8e32bad6-bd6d-4609-961d-3c1e558f49a2.png
https://user-images.githubusercontent.com/119288183/229331852-05d9c25b-8c33-4867-bca7-4a0ec6eaeb4d.png
https://user-images.githubusercontent.com/119288183/229331861-61de58db-6a82-44a5-bcc4-ebb229277edf.png
https://user-images.githubusercontent.com/119288183/229331870-b0b6431a-6a81-405e-8b9d-e51034138aae.png
https://user-images.githubusercontent.com/119288183/229331879-6ac35049-de53-4d71-84bb-c48a1a1d206c.png
https://user-images.githubusercontent.com/119288183/229331904-f024ec0e-ec5c-4311-b065-2d697700b7c8.png
https://user-images.githubusercontent.com/119288183/229331917-c771a0b0-287a-427e-8ecf-bcb95677e460.png

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
