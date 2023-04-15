from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time
from sklearn.manifold import TSNE

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# show the shape of dataset
print('CONJUNTO ORIGINAL: ')
print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_test.shape))
print('Y_test:  '  + str(y_test.shape))

# test the dataset
print(y_test[0])
plt.imshow(x_test[0])
plt.show()

# subset of dataset (0 and 8)
train_filter = np.where((y_train == 0) | (y_train == 8))
test_filter = np.where((y_test == 0) | (y_test == 8))
x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

# test the y_train : just 0 and 8
print(y_train)

# show the shape of dataset
print('SOLO 8 Y 0: ')
print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_test.shape))
print('Y_test:  '  + str(y_test.shape))
print(type(x_test))

# flat the dataset
num_pixels = x_train.shape[1] * x_train.shape[2] # find size of one-dimensional vector
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32') # flatten training images
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32') # flatten test images

# setup and run  regression with original features 

regresion = LogisticRegression()
start_time = time.time()
regresion.fit(x_train, y_train)
y_pred = regresion.predict(x_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("El tiempo de ejecución fue de ", elapsed_time, "segundos.")
print(f'presicion : {accuracy_score(y_test, y_pred, normalize=True)}')

# 1. 5 the processs never end.  Total number of iterations reach limit.

# NOW LETS SEE WITH PCA

# Feature scaling
sc = StandardScaler()
x_train_norm = sc.fit_transform(x_train)   # se estandariza y centra 
x_test_norm = sc.transform(x_test)

#Applying PCA function
pca = PCA(n_components=150)
Variables_Principales = pca.fit_transform(x_train_norm) #rap:  encntremos las comp ppales
# Transformarmos las componentes principales a partir de los datos
Variables_Principales_test = pca.fit_transform(x_test_norm)


fig = plt.figure(figsize=(12,8)) # Se define el tamaño de la figura en ancho y altura

plt.scatter(Variables_Principales[:,0], Variables_Principales[:,1], edgecolors="m")
plt.axhline(0, color="blue") # Elegir color de la linea horizontal de referencia
plt.title('Dos componentes del PCA') # Titulo de la gráfica
plt.xlabel('Primera Componente') # Etiqueta del eje x
plt.ylabel('Segunda Componente') # Etiqueta del eje y
plt.show() # Mostrar figura  rap:  se separan los comp y se ven...aqui es bueno que esten separadosa


Varianza = pca.explained_variance_ratio_ # Calculamos la tasa de varianza de las componentes y generamos las componentes principales
#rap  que tanta info entrgan.....variaza explicada...normalmente la primera comp entraga mucha info...aqui casi el 70%
fig = plt.figure(figsize=(15,8))
plt.bar(np.arange(len(Varianza)), Varianza)
plt.xlabel('Componentes Principales')
plt.ylabel('Tasa de Varianza')
plt.title('PCA')
plt.xlim(0, len(Varianza))
plt.show()

Importancia_Componentes = pd.DataFrame(Varianza)
Importancia_Componentes = Importancia_Componentes.T
print("Porcentaje de Varianza detectada para las primeras 150 componentes: ", format(100*Importancia_Componentes.loc[:,0:150].sum(axis = 1).values))

## 100 COMPONENTES  EXPLICAN EL 80% DE LA VARIANZA.


# setup regression with 100 components
regresion = LogisticRegression()
start_time = time.time()
regresion.fit(Variables_Principales, y_train)
y_pred_vp = regresion.predict(Variables_Principales_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("El tiempo de ejecución fue de ", elapsed_time, "segundos.")
print(f'presicion : {accuracy_score(y_test, y_pred_vp, normalize=True)}')


#USE TSNE for data visualization in 2 dimensions

t_sne = TSNE(n_components=2, random_state=42)
x_reduced = t_sne.fit_transform(x_train)
