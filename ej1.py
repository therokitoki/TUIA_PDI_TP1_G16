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

# ******************************************************************************************
# *                               Declaración de Funciones                                 *
# ******************************************************************************************

def localHistEQ(img: np.ndarray, width: int, height: int) -> np.ndarray:

    """
    Implementa la ecualización local del histograma. 

    Parámetros:
        img: Imagen a procesar (en escala de grises)
        width: Ancho de la ventana de procesamiento (entero, positivo, impar)
        height: Alto de la ventana de procesamiento (entero, positivo, impar)

    Retorno:
        np.ndarray: Imagen resultante luego de aplicar la ecualización local del histograma.    
    """
    # Validación de parámetros width y height (enteros, positivos e impares).
    if width < 0 or width % 2 == 0 or type(width) != int:
        raise ValueError('El parámetro "width" debe ser entero, positivo e impar.')
    if height < 0 or height % 2 == 0 or type(height) != int:
        raise ValueError('El parámetro "height" debe ser entero, positivo e impar.')

    # Validación de imagen en escala de grises (2 dimensiones).
    if len(img.shape) > 2:
        raise ValueError('La imagen proporcionada debe estar en escala de grises.')

    # Agregado de un borde replicado alrededor de la imagen para permitir el procesamiento de píxeles cercanos a los bordes.
    # Esto asegura que todos los píxeles puedan ser ecualizados sin perder información en los bordes.
    img_bordes = cv2.copyMakeBorder(img, width//2, width//2, height//2, height//2, cv2.BORDER_REPLICATE)

    # Inicialización de imagen vacía (de las mismas dimensiones que la imagen original) para almacenar el resultado del procesamiento.
    img_salida = np.empty(img.shape)

    # Recorrido de cada píxel de la imagen original.
    for i in range(img.shape[0]): #filas

        for j in range(img.shape[1]): #columnas
            
            # Se extrae una ventana local centrada en el píxel (i, j) de tamaño 'width x height' desde la imagen con bordes.
            ventana = img_bordes[i:i+width, j:j+height]

            # Se aplica ecualización global del histograma a la ventana local.
            hist_ventana = cv2.equalizeHist(ventana)

            # Se reemplaza el píxel (i, j) en la imagen de salida por el valor del píxel central de la ventana ecualizada.
            img_salida[i, j] = hist_ventana[width//2, height//2] 
    
    # Se retorna la imagen ecualizada.
    return img_salida



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


