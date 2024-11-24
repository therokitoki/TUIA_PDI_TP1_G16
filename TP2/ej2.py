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

# Se importan las librerías a utilizar
import cv2
import numpy as np
import matplotlib.pyplot as plt
from customlib import *

# Se inicializa una lista donde se almacenarán las imagenes procesadas
imgs_proc = []

# Se inicializa un contador para las imágenes que no se detectó correctamente la patente
not_detected = 0

# Se itera sobre las 13 imagenes contenidas en el directorio donde se encuentra el script
for i in range(1, 13):

    if i < 10:
        img_auto = f'./img0{i}.png'
    else:
        img_auto = f'./img{i}.png'
    
    # Se carga la imagen
    img = cv2.imread(img_auto)

    # Se genera un subplot donde del lado izquierdo estará la imagen original y en la derecha la imagen de salida
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Imagen {i} original')
    plt.axis('off')

    # Se inicializa la funcion que detectará la patente de la imagen actual
    img_final, status = matDetection(img, 50, 170.0, 20.0, 3.0, 1.5, 7)
    imgs_proc.append(img_final)

    plt.subplot(1, 2, 2, sharex=ax,sharey=ax)
    plt.imshow(img_final, cmap='gray')
    # Dependiendo de si la detección fue exitosa o no se imprime su respectivo título
    if status:
        plt.title(f'Imagen {i} procesada')
    else:
        plt.title('No se pudo detectar la patente')
        not_detected += 1

    plt.axis('off')
    plt.show()

# Resumen
plt.figure(figsize=(15, 12))
idx = 1

for img in imgs_proc:
    plt.subplot(4, 3, idx)
    plt.imshow(img, cmap='gray')
    plt.title(f'Imagen {idx} procesada')
    plt.axis('off')
    idx += 1
plt.tight_layout()
plt.show()

# Reporte final
detected = 12 - not_detected
print('Procesamiento de imágenes completo\nReporte final:\n')
print(f'Patentes detectadas: {detected}')
print(f'Porcentaje de éxito: {detected/12 * 100}%')
