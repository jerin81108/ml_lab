import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1,2,3,4,5,6,7,8,9,10])
Y = np.array([2,4,5,4,5,7,8,9,10,12])

# Calculate coefficients
n=len(X); Sx=np.sum(X); Sy=np.sum(Y); Sxy=np.sum(X*Y); Sx2=np.sum(X**2)
m=(n*Sxy - Sx*Sy)/(n*Sx2 - Sx**2)
b=(Sy - m*Sx)/n

print(f"Slope (m): {m:.4f}")
print(f"Intercept (b): {b:.4f}")
print(f"Equation: Y = {m:.4f}X + {b:.4f}")

# Predict
pred=m*X+b
ss_res=np.sum((Y-pred)**2); ss_tot=np.sum((Y-np.mean(Y))**2)
print(f"R² Score: {1 - ss_res/ss_tot:.4f}")

# Plot
plt.scatter(X,Y,color='blue',label='Actual')
plt.plot(X,pred,color='red',label='Predicted')
plt.xlabel('X'); plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend(); plt.grid(True); plt.show()
