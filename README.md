# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: v.sreeja
RegisterNumber:  212222230169
*/
```
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



## Output:
!![Screenshot (97)](https://user-images.githubusercontent.com/118344328/229777931-d781b95c-1d90-4ca5-9a10-fc4ad4652538.png)
[linear regression using gra
![Screenshot (98)](https://user-images.githubusercontent.com/118344328/229778076-a67bcff5-2b34-4c1e-876a-a24f1ca923e8.png)
dient descent](sam.png)
![Screenshot (99)![Screenshot (100)](https://user-images.githubusercontent.com/118344328/229778577-d3c313ff-9d66-4d7a-88c4-6e62dc63025d.png)
](https://user-images.githubusercontent.com/118344328/229778381-63661ace-c7e3-4e39-9bbd-9295d81f2f47.png)
![Screenshot (101)](https://user-images.githubusercontent.com/118344328/229778708-70a62f0d-b998-4ef3-9b84-0461b241c01b.png)
![Screenshot (102)](https://user-images.githubusercontent.com/118344328/229779042-0a7996e8-4cec-4543-8aae-256748a63e01.png)
![Uploading Screenshot (109).png…]()

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
