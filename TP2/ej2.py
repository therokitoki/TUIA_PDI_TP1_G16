############################################################################################
#                                                                                          #
#                              PROCESAMIENTO DE IMÁGENES 1                                 #
#                                 TRABAJO PRÁCTICO N°2                                     #
#                                                                                          #
#          GRUPO N°16: Gonzalo Asad, Sergio Castells, Agustín Alsop, Rocio Hachen          #
#                                                                                          #
#                           Problema 2 - Detección de patentes                             #
#                                                                                          #
############################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_auto = '.\img02.png'

#Cargo la imagen y la convierto a escala de grises
img = cv2.imread(img_auto)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1- Umbralado
_, thresh_img = cv2.threshold(gray, thresh=110, maxval=255, type=cv2.THRESH_BINARY)
plt.imshow(thresh_img, cmap='gray')
plt.show()

# 2- Componentes conectadas

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, 8, cv2.CV_32S)

plt.imshow(labels, cmap='gray')
plt.show()

label_filter = []
# 3- Filtrado por área
print(stats)
for i in range(len(stats)):
    if stats[i][4] > 30:
        label_filter.append(labels[i])
plt.imshow(labels[0], cmap='gray')
plt.show()
# 4- Filtrado por relación de aspecto

# 5- Componentes conectadas + BBOK + label