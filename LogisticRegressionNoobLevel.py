import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv (r'C:\Users\Shubhi Jain\Documents\Machine Learning tut\the-ultimate-halloween-candy-power-ranking\candy-data.csv')

x1 = df[['sugarpercent', 'pricepercent', 'winpercent']].values
x1df = pd.DataFrame(x1) #conversion to dataframe so that we can apply pandas operations
x = x1df.head(50) #to extract first 50 rows from x1 dataframe
m = len(x) #no of examples
y1 = pd.DataFrame(df['chocolate'].values)
y = y1.head(50) #extract first 50 chocolate values
w= np.random.rand(1,3)
b = 0 #initialize b and broadcasting will do the rest
z = np.dot(w, x.T) + b #for making x a vector with dimension 3*50, we have to take transpose
a = np.sign(z)
dz = a.T-y
#working
w = np.dot(x.T,dz)/m
b = np.sum(b)/m
#print(w)
#print(b)

#testing and calculating accuracy
tx = x1df.tail(36)
ty = y1.tail(36)
tz = np.dot(w.T, tx.T) +b
ta = np.sign(tz)
right, wrong = 0,0
s = ta.size
#print(ta.size)
#working
right = ty-ta.T
right = np.sum(right)
print(right)
accuracy = (36+right)/s * 100
print(accuracy)

