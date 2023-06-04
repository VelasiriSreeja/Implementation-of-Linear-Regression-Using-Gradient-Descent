# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import all the libraries which are needed to the program.
2.  get profit prediction graph and computecost value.
3. Get a graph of cost function using gradient descent and also get profit prediction graph.
4. Get the otput of profit for the population of 35,000 and 70,000.

 

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

![Screenshot (104)](https://user-images.githubusercontent.com/118344328/229823153-36212fa6-8069-440d-b836-d20b642ab25e.png)

![Screenshot (111](https://user-images.githubusercontent.com/118344328/229823569-450f19cf-e993-4396-8739-70175d4d9b26.png)

![Screenshot (112](https://user-images.githubusercontent.com/118344328/229823819-da872b9c-3288-4f54-9748-41c139856fc6.png)

![Screenshot (105)](https://user-images.githubusercontent.com/118344328/229823947-6db710f2-f6d2-4ce8-a053-b9e772e37359.png)


![Screenshot (106)](https://user-images.githubusercontent.com/118344328/229824156-2ac083be-8833-4c52-b5a3-e18e3d7bc490.png)

![Screenshot (113](https://user-images.githubusercontent.com/118344328/229824312-b8f4f90a-68f4-4d8b-bc7c-6c985aed8fd2.png)

![Screenshot (115](https://user-images.githubusercontent.com/118344328/229824438-f65432c6-a492-4fa9-9bd5-03f4676d4e31.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
