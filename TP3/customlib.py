############################################################################################
#                                                                                          #
#                              PROCESAMIENTO DE IMÁGENES 1                                 #
#                                 TRABAJO PRÁCTICO N°3                                     #
#                                                                                          #
#          GRUPO N°16: Gonzalo Asad, Sergio Castells, Agustín Alsop, Rocio Hachen          #
#                                                                                          #
#                                 Librería de Funciones                                    #
#                                                                                          #
############################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ******************************************************************************************
# *                               Declaración de Funciones                                 *
# ******************************************************************************************

def roiDetect(img: np.ndarray, percent: int=5, thresh: int=100, save: bool=False) -> tuple[int, int, int, int]:

    """
    Detecta una región específica en una imagen utilizando procesamiento de color y segmentación de componentes conectados,
    y retorna un rectángulo ajustado con un margen proporcional basado en un porcentaje especificado.

    Parámetros:
        frame: Imagen de entrada en formato BGR.
        percent: Porcentaje del margen a ajustar en los bordes del rectángulo detectado.
                    Debe ser un número entero entre 1 y 25. Por defecto, es 5%.
        thresh: Valor de umbral para obtener una máscara binaria.            
        save: Si se establece en 'True' guarda la imágen procesada en la carpeta `./frames`. 

    Retorno:
        Coordenadas del área detectada: (x_ini, x_fin, y_ini, y_fin).
 
    """
    # Validación del parámetro 'percentage'
    if not isinstance(percent, int) or not (1 <= percent <= 25):
        raise ValueError("El parámetro 'porc' debe ser un número entero entre 1 y 25.")
    
    # Validación del parámetro 'thresh'
    if not isinstance(thresh, int) or not (1 <= thresh <= 255):
        raise ValueError("El parámetro 'thresh' debe ser un número entero entre 1 y 255.")

    # frame_test = frame.copy()

    # Conviersión de la imagen de entrada de formato BGR a LAB.
    img_test = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Separación de los canales de la imagen LAB y umbralado del canal 'A'.
    L, A, B = cv2.split(img_test)

    _, thresh_img = cv2.threshold(A, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY_INV)

    # Detección de componentes conectadas.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, 8, cv2.CV_32S)

    print(stats)
    # Calcula de los vértices de la región detectada, considerando el márgen de ajuste.
    x,y,w,h,a = stats[1]

    x_ini = round(x+(w*(percent/100)))
    y_ini = round(y+(h*(percent/100)))
    x_fin = round(x+(w*(1 - percent/100)))
    y_fin = round(y+(h*(1 - percent/100)))

    # Opcional: Se guarda la imágen procesada en la carpeta `./frames`.
    if save:
        os.makedirs("./frames", exist_ok = True)  # Si no existe, crea la carpeta 'frames' en el directorio actual.

        cv2.rectangle(img, (x_ini, y_ini), (x_fin, y_fin), (0, 255, 0), 3)

        cv2.imwrite(os.path.join("./frames", f"Area_Detectada.jpg"), img)


    return (x_ini, x_fin, y_ini, y_fin)


# ******************************************************************************************
# ******************************************************************************************


def centroidsDetect(img: np.ndarray, th_min: int=1, min_area: int=0, max_area: int=1, jump: int=1) -> tuple[bool, list, list]:

    
    # Validación del parámetro 'max_area'
    img_area = img.shape[0] * img.shape[1] # Area de la imágen
    if not isinstance(max_area, int) or not (1 <= max_area <= img_area):
        raise ValueError("El parámetro 'max_area' debe ser un número entero menor al area de la imagen.")
            
    # Validación del parámetro 'min_area'
    if not isinstance(min_area, int) or not (1 <= min_area < max_area):
        raise ValueError("El parámetro 'mix_area' debe ser un número entero menor a 'max_area'.")
    
    # Validación del parámetro 'th_min'
    if not isinstance(th_min, int) or not (1 <= th_min <= 255):
        raise ValueError("El parámetro 'th_min' debe ser un número entero entre 1 y 255.")

            
    
    flag = False
    
    while not flag and th_min <= 120:
        _, thresh_img_a = cv2.threshold(img, thresh=th_min, maxval=255, type=cv2.THRESH_BINARY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img_a, 8, cv2.CV_32S)

        centroid_list = []

        if num_labels != 6:
            th_min += jump
            continue

        for i in range(len(stats)):
            x, y, w, h, a = stats[i]

            if a < max_area and a > min_area:
                centroid_list.append(centroids)

        # Si la cantidad de componentes identificadas es menor de 6, se descarta el umbral
        if len(centroid_list) != 5:
            th_min += jump
            continue
        
        #print(th_min)
        #print(stats)
        flag = True

    return flag, centroid_list, stats

# ******************************************************************************************
# ******************************************************************************************

def motionDetector(ant: list, act: list, thresh: int=5) -> bool:
# Acá hay que comparar los centroides anteriores y actuales y en caso de detectar 
# un desplazamiento inferior al umbral se considera que no existe movimiento.
    pass