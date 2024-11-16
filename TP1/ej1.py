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
from customlib import localHistEQ

# ******************************************************************************************
# *                                     Implementación                                     *
# ******************************************************************************************

# Carga y visualización de la imagen original en escala de grises.
# La imagen contiene detalles escondidos que se quieren resaltar mediante ecualización local del histograma.
img = cv2.imread('.\\img\\Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Imagen original en escala de grises')
plt.show()

# Se define una lista de diferentes tamaños de ventanas para la ecualización local.
# Cada par (ancho, alto) representa el tamaño de la ventana de procesamiento que se aplicará a la imagen.
win_sizes = [(3,3),(5,5),(11,11),(21,11),(35,25),(51,51),(101,101)]

# Se aplica ecualización local del histograma con ventanas de distintos tamaños.
plt.figure(figsize=(14,8))
ax = plt.subplot(2,4,1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original')

for i, (w, h) in enumerate(win_sizes):
    img_eq = localHistEQ(img, w, h)
    plt.subplot(2, 4, i+2,sharex=ax,sharey=ax) # 'sharex' y 'sharey' permiten compartir los ejes con la imagen original para mantener la escala.
    plt.imshow(img_eq,cmap='gray')
    plt.title(f'Ventana {w}x{h}')

# Se muestran todas las imágenes procesadas.  
plt.show()


