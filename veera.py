import numpy as np

X = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
y = np.array([[1,0],[1,0],[0,1],[0,1]])

sig = lambda x: 1 / (1 + np.exp(-x))
sig_d = lambda x: x * (1 - x)

np.random.seed(0)
W1 = np.random.uniform(size=(4, 5))
W2 = np.random.uniform(size=(5, 2))

for _ in range(10000):
    h = sig(X @ W1)
    o = sig(h @ W2)
    d2 = (y - o) * sig_d(o)
    d1 = (d2 @ W2.T) * sig_d(h)
    W2 += h.T @ d2 * 0.1
    W1 += X.T @ d1 * 0.1

predict = lambda v: sig(sig(v @ W1) @ W2)

print("cat:", predict(np.array([1,0,0,0])))
print("dog:", predict(np.array([0,1,0,0])))
print("run:", predict(np.array([0,0,1,0])))
print("eat:", predict(np.array([0,0,0,1])))