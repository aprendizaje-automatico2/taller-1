from unsupervised.dimensionality_reduction.funciones_cadena import *
# from unsupervised.dimensionality_reduction.svd import *
from scipy.linalg import svd
import matplotlib.pyplot as plt
import matplotlib.image as img
# from PIL import Image
import cv2
import numpy as np

#test with library
print(contar_letras('holamundo'))

#open the image
img = cv2.imread('C:/Users/robinsonap/Downloads/fotos/robinson_alvarez.jpeg')
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#image 256x256 
img256 = cv2.resize(gray1, (256, 256), interpolation=cv2.INTER_NEAREST)
cv2.imshow('face', img256)
cv2.waitKey(0)
print('Imagen: ' , img256)

# print(type(imgmat))
# print(type(img256))
# print(imgmat)

# calculate Singular Value Decomposition in order image compression
U, S, Vt = np.linalg.svd(img256)
print(type(U))
print(U)

#computing the first column of U and the first raw of V reproduce the most prominent feature of the image

reconstimg = U[:, :1] @ np.diag(S[:1]) @ Vt[:1, :]
#print(reconstimg)
img_n = cv2.normalize(src=reconstimg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('facemedio', img_n)
cv2.waitKey(0)
print(img_n.shape)
print(type(img_n))
# Imprimo la imagen