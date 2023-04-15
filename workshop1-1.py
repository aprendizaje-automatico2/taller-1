print("\033[4m \n\n 1. Work with matrix (rank, trace, invert and eagenvalues): \033[0m \n\n")
import numpy as np
import random
np.set_printoptions(precision = 2,linewidth=80)
# Creacion de la matriz random
A =  np.random.randint(0, 100, size=(random.randint(2,2),random.randint(2,2)))   
print(f'Matriz inicial A: \n {A}')
print(f"\nRange of Matrix A : \n {np.linalg.matrix_rank(A)}")                      # RANGE
print(f"\nTrace of Matrix A: \n {np.trace(A)}")  

# validar si es cuadrada para el determinante
if A.shape[0] == A.shape[1] :
    detA = np.linalg.det(A)                                                     # DETERMINANT detA
    print(f"\nDeterminant of A: \n {round(detA)}\n")
else:
    print("\nMatriz is not square.  Unable to set deteterminant")

try: # validar exception si es invertible
    invA = np.linalg.inv(A)                                                     # INVERSE invA
    print(f"\nInverse of A: \n {np.round(invA, 3)} \n")  
except :
    print("\nMatrix has not Inverse\n")

# eigenvalues y eigenvectores
w,v = np.linalg.eig(A)                                                          # EAGENVALUES and EAGENVECTORS OF A : w, v
print(f"\neigenvalues of A: \n {np.round(w,3)}\n")
print(f"\neigenvectors of A: \n {np.round(v,3)}\n")
wi,vi = np.linalg.eig(invA)
print(f"\neigenvalues of A inverse: \n {np.round(wi,3)}\n")                                      # EAGENS OF Inverse
print(f"\neigenvectors of A inverse:\n {np.round(vi,3)}\n")
I = np.matmul(A,invA)
print(f"\nPRODUCT OF A'A: \n {np.round(I,3)}")                                     # A'A = I
print('\nResult is Identity matrix\n')
print('\nEAGENVALUES AND EAGEN VECTORS FOR IDENTITY MATRIX IS:  AxAinv\n  ')
print(np.linalg.eig(A@invA))
print('\nAhora AinvxA: \n')
print(np.linalg.eig(invA@A))   # eagenalues are the same (1), but the eagenvector are diferents