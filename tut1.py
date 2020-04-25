#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
data = pd.read_csv(".\headbrain.csv")
print(data.shape)
print(data.head())

#collection x and y
X= data['Head Size(cm^3)'].values
Y= data['Brain Weight(grams)'].values

mean_x = np.mean(X) #mean values
mean_y = np.mean(Y)

n = len(X)  #total number of values

numer = 0
denom = 0

for i in range(n):
	numer += (X[i]- mean_x)*(Y[i]-mean_y)
	denom += (X[i]-mean_x)**2

b0 = numer/denom
b1 = mean_y - (b0*mean_x)

print(b0, b1)

# Plotting Values and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100
# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b1 + b0 * x 
 
# Ploting Line
plt.plot(x, y, color='#52b920', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef4423', label='Scatter Plot')
 
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

#ss_t is the total sum of squares and ss_r is the total sum of squares of residuals(relate them to the formula).
ss_t = 0
ss_r = 0
for i in range(n):
	y_pred = b1 + b0 * X[i]
	ss_t += (Y[i] - mean_y) ** 2
	ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)