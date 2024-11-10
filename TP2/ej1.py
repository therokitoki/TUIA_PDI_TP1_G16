############################################################################################
#                                                                                          #
#                              PROCESAMIENTO DE IMÁGENES 1                                 #
#                                 TRABAJO PRÁCTICO N°1                                     #
#                                                                                          #
#          GRUPO N°16: Gonzalo Asad, Sergio Castells, Agustín Alsop, Rocio Hachen          #                                                                                          
#                                                                                          #
#                       Problema 1 - Ecualización local de histograma                      #
#                                                                                          #
############################################################################################ 

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_moneda = 'TP2\monedas.jpg'

img = cv2.imread('TP2\monedas.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Imagen original en escala de grises')
plt.show()

bright = cv2.imread(img_moneda)
bright = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)


# # Transformamos a CIELAB

brightLAB = cv2.cvtColor(bright, cv2.COLOR_RGB2LAB)

plt.figure()
ax1 = plt.subplot(141); plt.xticks([]), plt.yticks([]), plt.imshow(bright), plt.title('BRIGHT')
plt.subplot(142,sharex=ax1,sharey=ax1), plt.imshow(brightLAB[:,:,0], cmap="gray"), plt.title('L')
plt.subplot(143,sharex=ax1,sharey=ax1), plt.imshow(brightLAB[:,:,1], cmap="gray"), plt.title('A')
plt.subplot(144,sharex=ax1,sharey=ax1), plt.imshow(brightLAB[:,:,2], cmap="gray"), plt.title('B')
plt.show()

# # Transformamos a HSV

# bright_HSV = cv2.cvtColor(bright, cv2.COLOR_RGB2HSV)

# plt.figure()
# ax1 = plt.subplot(141); plt.xticks([]), plt.yticks([]), plt.imshow(bright), plt.title('BRIGHT')
# plt.subplot(142,sharex=ax1,sharey=ax1), plt.imshow(bright_HSV[:,:,0], cmap="gray"), plt.title('H')
# plt.subplot(143,sharex=ax1,sharey=ax1), plt.imshow(bright_HSV[:,:,1], cmap="gray"), plt.title('S')
# plt.subplot(144,sharex=ax1,sharey=ax1), plt.imshow(bright_HSV[:,:,2], cmap="gray"), plt.title('V')

# # plt.show()

# bright_RGB = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)

# ax1 = plt.subplot(141); plt.xticks([]), plt.yticks([]), plt.imshow(bright), plt.title('BRIGHT')
# plt.subplot(142,sharex=ax1,sharey=ax1), plt.imshow(bright_RGB[:,:,0], cmap="gray"), plt.title('R')
# plt.subplot(143,sharex=ax1,sharey=ax1), plt.imshow(bright_RGB[:,:,1], cmap="gray"), plt.title('G')
# plt.subplot(144,sharex=ax1,sharey=ax1), plt.imshow(bright_RGB[:,:,2], cmap="gray"), plt.title('B')
# plt.show()

############## Codigo con umbralado    
#Cargar la imagen
img = cv2.imread(img_moneda)

#Convertir a escala de grises
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = brightLAB[:,:,2]


#Aplicar umbralización para binarizar la imagen
thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)[1]
plt.imshow(thresh, cmap='gray')
plt.show()

#Encontrar los contornos
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Dibujar los contornos
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
monedas = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)

    #print(area, perimetro)
    if perimetro < 50 or area < 1000:
        continue
    factor = area / (perimetro ** 2)

    if factor > 0.01:
        monedas.append(factor)
        print(1/factor)
#Mostrar la imagen resultante
print(len(monedas))
plt.imshow(img)
plt.show()

############## Codigo con Canny 
img = cv2.imread(img_moneda)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray)
plt.show()
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

plt.imshow(blurred)
plt.show()

edges = cv2.Canny(blurred, 30, 100)

plt.imshow(edges)
plt.show()

#Encontrar los contornos
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Dibujar los contornos
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
monedas = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)

    #print(area, perimetro)
    if perimetro < 50 or area < 1000:
        continue
    factor = area / (perimetro ** 2)

    if factor > 0.01:
        monedas.append(factor)
        print(1/factor)
#Mostrar la imagen resultante
print(len(monedas))
plt.imshow(img)
plt.show()