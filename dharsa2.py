A=[[1,2,3],[4,5,6],[7,8,9]]
B=[[9,8,7],[6,5,4],[3,2,1]]

R=[[A[i][j]+B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

print("Matrix A:"); [print(r) for r in A]
print("Matrix B:"); [print(r) for r in B]
print("Sum A+B:"); [print(r) for r in R]
