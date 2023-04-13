print("ITEM 1 : \n\n")
import numpy as np
import random
np.set_printoptions(precision = 2,linewidth=80)
# Creacion de la matriz
A =  np.random.randint(0, 100, size=(random.randint(2,2),random.randint(2,2)))   
print(A)
print(f"\nRange of Matrix A : {np.linalg.matrix_rank(A)}")                      # RANGE
print(f"\nTrace of Matrix A: {np.trace(A)}")                                    # TRACE
if A.shape[0] == A.shape[1] :
    detA = np.linalg.det(A)                                                     # DETERMINANT detA
    print(f"\nDeterminant of A: {round(detA)}\n")
else:
    print("\nMatriz is not square.  Unable to set deteterminant")
try:
    invA = np.linalg.inv(A)                                                     # INVERSE invA
    print(f"\nInverse of A: \n {np.round(invA, 3)}")  
except :
    print("\nMatrix has not Inverse\n")
w,v = np.linalg.eig(A)                                                          # EAGENVALUES and EAGENVECTORS OF A : w, v
print(f"eigenvalues of A: {np.round(w,3)}\n")
print(f"eigenvectors of A: {np.round(v,3)}\n")
wi,vi = np.linalg.eig(invA)
print(f"eigenvalues of A inverse: {np.round(wi,3)}\n")                                      # EAGENS OF Inverse
print(f"eigenvectors of A inverse: {np.round(vi,3)}\n")
I = np.matmul(A,invA)
print(f"\nPRODUCT OF A'A: {np.round(I,3)}")                                     # A'A = I
print('\nResult is Identity matrix\n')
print('EAGENVALUES AND EAGEN VECTORS FOR IDENTITY MATRIX IS:  ')
print(np.linalg.eig(A@invA))
print('al revez: ')
print(np.linalg.eig(invA@A))   # eagenalues are the same (1), but the eagenvector are diferents