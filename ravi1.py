# ravi1.py - Candidate Elimination Algorithm

import csv

# Load CSV data
with open('training_data.csv') as f:
    data = list(csv.reader(f))

attributes = data[0][:-1]   # Feature names
examples   = data[1:]        # Training rows

# Initialize S (most specific) and G (most general)
S = ['0'] * len(attributes)
G = [['?'] * len(attributes)]

for row in examples:
    X, label = row[:-1], row[-1]
    if label == 'Yes':
        G = [g for g in G if all(g[i]=='?' or g[i]==X[i] for i in range(len(X)))]
        S = [X[i] if S[i]=='0' else ('?' if S[i]!=X[i] else S[i]) for i in range(len(X))]
    else:
        G = [h for g in G for i in range(len(X)) if g[i]=='?' and X[i]!=S[i]
             for h in [g[:i]+[S[i]]+g[i+1:]]]

print("Final S:", S)
print("Final G:", G)
