from PIL import Image
import cv2
import numpy as np
# open the image in gray 
img1 = cv2.imread('C:/Users/robinsonap/Downloads/fotos/robinson_alvarez.jpeg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('C:/Users/robinsonap/Downloads/fotos/AlejandroC.jpeg',cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('C:/Users/robinsonap/Downloads/fotos/JuanPabloM.png',cv2.IMREAD_GRAYSCALE)

#change the size
dsize = (256, 256)
output1 = cv2.resize(img1, dsize, interpolation=cv2.INTER_NEAREST)
output2 = cv2.resize(img2, dsize, interpolation=cv2.INTER_NEAREST)
output3 = cv2.resize(img3, dsize, interpolation=cv2.INTER_NEAREST)

# mean of images

avg1 = cv2.addWeighted(output1, 0.5, output2, 0.5, 0)
prom = cv2.addWeighted(avg1, 0.5, output3, 0.5, 0)

# resultado de la mescla de imagenes
cv2.imshow('Mescla',prom)
cv2.waitKey(0)

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces_rect = haar_cascade.detectMultiScale(output1, scaleFactor=1.1, minNeighbors=9)
print(type(faces_rect))
print(faces_rect[0][0])
cv2.rectangle(output2,(faces_rect[0][0],faces_rect[0][1]), (faces_rect[0][0]+faces_rect[0][2], faces_rect[0][1]+faces_rect[0][3]),(0,255,0), thickness=2 )
cv2.imshow('hola', output2)
cv2.waitKey(0)

# 1. CALCULO DE LA DISTANCIA
# 1.1 Distancia euclidiana (raiz caudrada de la suma de los cuadrados de las diferencias)
distance = np.sqrt(np.sum((output1 - prom)**2))
print('La distancia Eucliadian es: ', distance)


# 1.2 Diferencia de Histogramas
hist1 = cv2.calcHist([output1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([prom], [0], None, [256], [0, 256])

# normalizo histogramas
cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

#calcular distancia de Bhattacharyya (mide el nivel de superposicion entre dos distribuciones, entre 0 y 1. 0 identicas)
distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

# Mostrar la distancia
print('La distancia de Bhattacharyya entre los histogramas es: ', distance)


