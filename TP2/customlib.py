############################################################################################
#                                                                                          #
#                              PROCESAMIENTO DE IMÁGENES 1                                 #
#                                 TRABAJO PRÁCTICO N°2                                     #
#                                                                                          #
#          GRUPO N°16: Gonzalo Asad, Sergio Castells, Agustín Alsop, Rocio Hachen          #                                                                                          
#                                                                                          #
#                                 Librería de Funciones                                    #
#                                                                                          #
############################################################################################ 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ******************************************************************************************
# *                               Declaración de Funciones                                 *
# ******************************************************************************************

def pltimg(img: np.ndarray, cmap: str, title: str):
    """
    Realiza un imshow de la imagen adjuntada

    Parámetros:
        img: Imagen a mostrar
        cmap: Mapeo de la imagen a mostrar
        title: Título de la imgagen a mostrar

    Retorno:

    """
    plt.imshow(img, cmap=f'{cmap}')
    plt.title(title)
    plt.show()

def matDetection(img: np.ndarray, th_min: int, max_area: float, min_area: float, max_aspect_ratio: float, min_aspect_ratio: float) -> np.ndarray:
    """
    Detecta en una imagen dada la patente y los caracteres que la compone.

    Parámetros:
        img: Imagen a procesar (en escala de grises)
        th_min: Valor de threshold donde empezará a iterar (entero, positivo)
        max_area: Máxima area de la letra a detectar (entero, positivo)
        min_area: Mínima area de la letra a detectar (entero, positivo)
        max_aspect_ratio: Máximo ratio de aspecto (entero, positivo)
        min_aspect_ratio: Mínimo ratio de aspecto (entero, positivo)

    Retorno:
        np.ndarray: Imagen resultante luego de detectar y remarcar la patente en el caso de haber sido detectado y sus respectivos caracteres que las componen.
    """
    # Se convierte en escala de grises la imagen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #pltimg(gray, "gray", "Imagen escala de grises")

    # Se define un estado de la imagen
    img_valida = False

    # Se genera un loop que se detendrá cuando se detecte correctamente la patente o cuando alcanza el máximo de los hiperparametros
    while not img_valida and th_min <= 250:

        # Se realiza una copia de la imagen
        img_final = img.copy()

        # Para la detección de la patente se sigue los siguientes 3 pasos
        # 1- Umbralado
        _, thresh_img = cv2.threshold(gray, thresh=th_min, maxval=255, type=cv2.THRESH_BINARY)

        #pltimg(thresh_img, "gray", "Umbralado")


        # 2- Componentes conectadas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, 8, cv2.CV_32S)

        #pltimg(labels, "gray", "Componentes conectadas")


        # 3- Filtrado por área
        filtered_labels = labels.copy()
        filtered_labels = filtered_labels.astype("uint8")
        filtered_letters = labels.copy()
        filtered_letters = filtered_letters.astype("uint8")

        filtered_index = []
        letters_index = []

        for i in range(len(stats)):
            x, y, w, h, a = stats[i]

            if a < max_area and a > min_area:
                filtered_index.append(i)

                # 4- Filtrado por relación de aspecto
                ar = h / w
                if ar >= min_aspect_ratio and ar <= max_aspect_ratio:
                    letters_index.append(i)

                    # 5- Label
                    cv2.rectangle(img_final, (x, y), (x+w, y+h), (0, 0, 255), 1)

        if len(letters_index) != 6:
            #print(len(letters_index),th)
            th_min += 1
            continue

        coord_x = []
        coord_y = []
        alturas = []

        for i in letters_index:
            x, y, w, h, a = stats[i]
            coord_x.append(int(x))
            coord_y.append(int(y))
            alturas.append(int(h))

        coord_x.sort(),coord_y.sort(),alturas.sort()

        x_ant = coord_x[0]
        y_ant = coord_y[0]
        maximo_x = 0
        maximo_y = 0

        for x in coord_x:
            dist = x-x_ant
            if maximo_x < dist:
                maximo_x = dist
            x_ant = x

        for y in coord_y:
            dist = y-y_ant
            if maximo_y < dist:
                maximo_y = dist
            y_ant = y

        #print(maximo_x, 'maximo x')
        #print(maximo_y, 'maximo y')

        if maximo_y > 5 or maximo_x > 25:
            th_min += 1
            continue

        cv2.rectangle(img_final, (coord_x[0]-5, coord_y[0]-round((alturas[5]//2)*1.5)), (coord_x[5]+round((coord_x[5]-coord_x[4])*1.5), coord_y[0]+(alturas[5]*2)), (0, 255, 0), 1)

        # Imprimir labels filtrados
        mask_filtered_labels = np.isin(filtered_labels, filtered_index)

        filtered_labels[mask_filtered_labels] = 255
        filtered_labels[~mask_filtered_labels] = 0

        #pltimg(filtered_labels, "gray", "Filtrado por área")

        # Imprimir letras filtradas
        mask_filtered_letters = np.isin(filtered_letters, letters_index)

        filtered_letters[mask_filtered_letters] = 255
        filtered_letters[~mask_filtered_letters] = 0

        #pltimg(filtered_letters, "gray", "Filtrado por relación de aspecto")

        img_valida = True
        #print(th)

    print(f"Umbral final: {th_min}")
    return cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), img_valida