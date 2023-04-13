import numpy as np
from scipy.linalg import svd

# La función svd_reducida implementa la descomposición en 
# valores singulares reducida utilizando la matriz de covarianza. 
# Primero, calcula la matriz de covarianza C como la multiplicación de la matriz A transpuesta por A. 
# Luego, obtiene los valores y vectores propios de C y los ordena de forma inversa. 
# A partir de estos valores y vectores propios, se calculan las matrices U, S y Vt

def svd_reducida(A):
    """
    Descomposición en valores singulares reducida de la matriz A.
    Devuelve las matrices U, S, Vt.
    """
    # Calcula la matriz de covarianza
    C = np.dot(A.T, A)

    # Obtiene los valores y vectores propios de la matriz de covarianza
    # La función eig devuelve los valores y vectores propios en orden ascendente,
    # por lo que es necesario ordenarlos de forma inversa.
    eigvals, eigvecs = np.linalg.eig(C)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    print(eigvals)

    # Obtiene la matriz U
    U = np.dot(A, eigvecs)
    U = U / np.linalg.norm(U, axis=0)

    # Obtiene los valores singulares y la matriz Vt
    S = np.sqrt(eigvals)
    Vt = eigvecs.T

    return U, S, Vt

# Ejemplo de uso
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = svd_reducida(A)
print("U:\n", U)
print("S:\n", S)
print("Vt:\n", Vt)

# Comprobación
U_, S_, Vt_ = svd(A)
print("U_:\n", U_)
print("S_:\n", S_)
print("Vt_:\n", Vt_)
