import numpy as np

# Dataset: AND gate
X=np.array([[0,0],[0,1],[1,0],[1,1]]); Y=np.array([0,0,0,1])

# Init
w=np.zeros(X.shape[1]); b=0; lr=0.1; epochs=10

# Train
for e in range(epochs):
    for i in range(len(X)):
        pred=1 if np.dot(X[i],w)+b>=0.5 else 0
        err=Y[i]-pred
        w+=lr*err*X[i]; b+=lr*err
    print(f"Epoch {e+1}: w={w}, b={b:.2f}")

# Test
print("\nResults:")
for i in range(len(X)):
    out=1 if np.dot(X[i],w)+b>=0.5 else 0
    print(f"Input:{X[i]} -> Predicted:{out} | Actual:{Y[i]}")
