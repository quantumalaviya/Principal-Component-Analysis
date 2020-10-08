import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
#Example X
X = np.array([[1, 2, 3, 7, 5],
              [4, 5, 6, 7, 1],
              [7, 8, 9, 9, 8]])
'''

m = int(input("Number of rows: ")) 
X = []

for _ in range(m):
    row = list(map(int, input().split(" ")))
    X.append(row)


#scaling makes X a singular matrix, however SVD will exist.
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#performing SVD
U, S, V = np.linalg.svd(X)

#minimum value of l; l can be any value greater than what this snippet returns (upto n)
l = X.shape[1]

while l > 0:
    if np.sum(S[:l]) < 0.99*np.sum(S):
        l+=1
        break
    l-=1

D = V[:l].T 

#encode
C = np.dot(X, D)

#decode
X_decode = np.dot(C, D.T)

#scale back to original data
X_decode = scaler.inverse_transform(X_decode)

print("Encoding...")
print(C)
print("Decoding...")
print(X_decode)
