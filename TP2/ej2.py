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

# 3- Filtrado por área
filtered_labels = labels.copy()
filtered_labels = filtered_labels.astype("uint8")

index_list = []
for i in range(len(stats)):
    area = stats[i][4]
    if area < 170 and area > 30:
        
        # 4- Filtrado por relación de aspecto
        ar = stats[i][3] / stats[i][2]
        if ar >= 1.5 and ar <= 3.0:
            index_list.append(i)

mask = np.isin(filtered_labels, index_list)

filtered_labels[mask] = 255
filtered_labels[~mask] = 0

plt.imshow(filtered_labels, cmap='gray')
plt.show()

# 5- Componentes conectadas + BBOK + label
img_final = img.copy()

for i in index_list:
    x, y, w, h, a = stats[i]
    cv2.rectangle(img_final, (x, y), (x+w, y+h), (255, 0, 0), 1)

plt.imshow(img_final, cmap='gray')
plt.show()