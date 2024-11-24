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
        title: Título de la imagen a mostrar

    Retorno:

    """
    if cmap == None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=f'{cmap}')
    plt.title(title)
    plt.show()


def slice_when(predicate, iterable):
  """
    Divide un iterable en sublistas basándose en una condición especificada por el predicado.

    Parámetros:
        predicate: Una función que toma dos argumentos (elementos consecutivos del iterable)
                   y devuelve True si se debe realizar una división en ese punto.
        iterable: Un iterable que se dividirá en sublistas.

    Retorno:
        list: Sublistas del iterable original, divididas según la condición del predicado.
    """
  i, x, size = 0, 0, len(iterable)
  while i < size-1:
    if predicate(iterable[i][0], iterable[i+1][0]): 
      yield iterable[x:i+1]
      x = i + 1
    i += 1
  yield iterable[x:size]

# tst = [1,3,4,6,8,22,24,25,26,67,68,70,72]
# slices = slice_when(lambda x,y: y - x > 2, tst)
# print(list(slices))
# #=> [[1, 3, 4, 6, 8], [22, 24, 25, 26], [67, 68, 70, 72]]


def matDetection(img: np.ndarray, th_min: int, max_area: float, min_area: float, max_aspect_ratio: float, min_aspect_ratio: float, jump: int = 1) -> np.ndarray:
    """
    Detecta en una imagen dada la patente y los caracteres que la compone.

    Parámetros:
        img: Imagen a procesar (en escala de grises)
        th_min: Valor de threshold donde empezará a iterar (entero, positivo)
        max_area: Máxima area de la letra a detectar (entero, positivo)
        min_area: Mínima area de la letra a detectar (entero, positivo)
        max_aspect_ratio: Máximo ratio de aspecto (entero, positivo)
        min_aspect_ratio: Mínimo ratio de aspecto (entero, positivo)
        jump: Incremento en el valor del umbral tras cada iteración

    Retorno:
        np.ndarray: Imagen resultante luego de detectar y remarcar la patente en el caso de haber sido detectado y sus respectivos caracteres que las componen.
        status: Un valor True indica que el procesamiento ha sido realizado exitosamente.
    """
    # Se convierte en escala de grises la imagen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #pltimg(gray, "gray", "Imagen escala de grises")

    # Se define un estado de la imagen
    status = False

    # Se genera un loop que se detendrá cuando se detecte correctamente la patente o cuando alcanza el máximo de los hiperparametros
    while not status and th_min <= 250:

        # Se realiza una copia de la imagen
        img_final = img.copy()

        #th_min = 143
        # Para la detección de la patente se sigue los siguientes pasos
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

                    # Almacenado de etiquetas
                    letters_index.append(i)

        # Si la cantidad de componentes identificadas es menor de 6, se descarta el umbral
        if len(letters_index) < 6:
            th_min += jump
            continue
        
        # Inicializo las listas donde guardaremos las coordenadas de las componentes identificadas
        coord_x = []
        coord_y = []

        for index in letters_index:
            x, y, w, h, a = stats[index]
            coord_x.append((int(x),index))
            coord_y.append((int(y),index))

        # Se ordena la lista de tuplas con formato (coordenada, label) según el valor de la coordenada.
        coord_x = sorted(coord_x, key=lambda x: x[0])
        coord_y = sorted(coord_y, key=lambda x: x[0])
        
        # Se agrupan las coordenadas x según su distancia
        slices_x = list(slice_when(lambda x1,x2: x2 - x1 > 30, coord_x))
        
        candidates = []
        found = False

        # Se identifican aquellas agrupaciones que contienen 6 elementos
        if slices_x != []:
            for slice in slices_x:
                if len(slice) == 6:
                    candidates.append(slice)
                    found = True
        
        if not found:
            th_min +=jump
            continue
        

        found = False

        for candidate in candidates:

            coord_y_filtered = []
            candidate_labels  = []

            # Se guardan los labels en una lista aparte
            for tuple in candidate:    
                candidate_labels.append(tuple[1])

            # Se filtran las coordenadas Y obtenidas según los labels de candidate_labels
            for tuple in coord_y:
                if tuple[1] in candidate_labels:
                    coord_y_filtered.append(tuple)

            # Validación de distancia en el eje Y
            y_ant = coord_y_filtered[0][0]

            maximo_y = 0

            for y in coord_y_filtered:
                dist = y[0]-y_ant
                if maximo_y < dist:
                    maximo_y = dist
                y_ant = y[0]

            if maximo_y < 5:
                found = True
                break        

        if not found:
            th_min += jump
            continue

        # Se grafican los rectángulos contenedores
        coord_x = []
        coord_y = []
        alturas = []
        
        for i in candidate_labels:
            x, y, w, h, a = stats[i]
            cv2.rectangle(img_final, (x, y), (x+w, y+h), (0, 0, 255), 1)
            coord_x.append(int(x))
            coord_y.append(int(y))
            alturas.append(int(h))

        coord_x.sort(),coord_y.sort(),alturas.sort()
        cv2.rectangle(img_final, (coord_x[0]-round((w*0.9)), coord_y[0]-round((alturas[5]//2)*1.5)), (coord_x[5]+round((coord_x[5]-coord_x[4])*1.5), coord_y[0]+round((alturas[5]*2))), (0, 255, 0), 1)

        # # Imprimir labels filtrados
        # mask_filtered_labels = np.isin(filtered_labels, filtered_index)

        # filtered_labels[mask_filtered_labels] = 255
        # filtered_labels[~mask_filtered_labels] = 0

        # #pltimg(filtered_labels, "gray", "Filtrado por área")

        # # Imprimir letras filtradas
        # mask_filtered_letters = np.isin(filtered_letters, letters_index)

        # filtered_letters[mask_filtered_letters] = 255
        # filtered_letters[~mask_filtered_letters] = 0

        # # #pltimg(filtered_letters, "gray", "Filtrado por relación de aspecto")

        status = True
        #print(th)

    print(f"Umbral final: {th_min}")
    return cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), status