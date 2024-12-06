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
    
    # Validación del parámetro 'save'
    if not isinstance(save, bool):
        raise ValueError("El parámetro 'save' debe ser un valor booleano (True o False).")

    # frame_test = frame.copy()

    # Conviersión de la imagen de entrada de formato BGR a LAB.
    img_test = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Separación de los canales de la imagen LAB y umbralado del canal 'A'.
    L, A, B = cv2.split(img_test)

    _, thresh_img = cv2.threshold(A, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY_INV)

    # Detección de componentes conectadas.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, 8, cv2.CV_32S)

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
    """
    Devuelve los centroides y estadísticas de determinadas componentes conectadas detectadas en una imágen. 
    
    Parámetros:
        img: Imágen de entrada.
        th_min: Valor de umbral inicial para obtener una máscara binaria.
        min_area: Área mínima que debe tener una componente conectada para ser considerada válida.
        max_area: Área máxima que puede tener una componente conectada para ser considerada válida. Debe ser menor o igual al área total de la imagen.
        jump: Incremento en el umbral 'th_min' en cada iteración si las condiciones no se cumplen.

    Retorno:
        flag: Indica si se encontraron exactamente 6 componentes válidas bajo las restricciones definidas.
        centroid_list: Lista con las coordenadas de los centroides de las componentes detectadas. 
        stats: Estadísticas de las componentes conectadas detectadas. Cada componente incluye:
            - Coordenada x superior izquierda.
            - Coordenada y superior izquierda.
            - Ancho.
            - Alto.
            - Área.
    """
    
    # Validación del parámetro 'th_min'
    if not isinstance(th_min, int) or not (1 <= th_min <= 255):
        raise ValueError("El parámetro 'th_min' debe ser un número entero entre 1 y 255.")
    
    # Validación del parámetro 'max_area'
    img_area = img.shape[0] * img.shape[1] # Area de la imágen
    if not isinstance(max_area, int) or not (1 <= max_area <= img_area):
        raise ValueError("El parámetro 'max_area' debe ser un número entero menor al area de la imagen.")
            
    # Validación del parámetro 'min_area'
    if not isinstance(min_area, int) or not (1 <= min_area < max_area):
        raise ValueError("El parámetro 'mix_area' debe ser un número entero menor a 'max_area'.")
    
    # Validación del parámetro 'jump'
    if not isinstance(jump, int) or not (1 <= jump <= 255):
        raise ValueError("El parámetro 'jump' debe ser un número entero entre 1 y 255.")
       
    flag = False
    
    while not flag and th_min <= 160:
        # Umbralado de la imagen de entrada
        _, thresh_img_a = cv2.threshold(img, thresh=th_min, maxval=255, type=cv2.THRESH_BINARY)

        # Detección de componentes conectadas.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img_a, 8, cv2.CV_32S)

        centroid_list = []
        count = 0

        # Verificación cantidad de labels identificadas
        if num_labels != 6:
            th_min += jump
            continue

        for i in range(len(stats)):
            x, y, w, h, a = stats[i]
            # Filtrado de labels por área
            if a < max_area and a > min_area:
                count += 1
                

        # Si la cantidad de labels filtradas es distinta de 5, se descarta el umbral
        if count != 5:
            th_min += jump
            continue
        
        #print(th_min)
        #print(stats)
        flag = True
        centroid_list = centroids.tolist()
    
    return flag, centroid_list, stats

# ******************************************************************************************
# ******************************************************************************************

def motionDetector(ant: list, act: list, thresh: float=5) -> bool:
    """
    Detecta si hay movimiento basado en el desplazamiento de los centroides.

    Esta función compara las posiciones de los centroides de dos listas consecutivas
    (anteriores y actuales) y determina si existe movimiento. Se considera que no hay 
    movimiento si el desplazamiento entre los centroides correspondientes de ambas listas 
    es menor a un umbral definido.

    Parámetros:
        ant: Lista de coordenadas (x, y) de los centroides en el estado anterior.
        act: Lista de coordenadas (x, y) de los centroides en el estado actual.
        thresh: Umbral de desplazamiento. Si el desplazamiento es menor
                que este valor, no se considera movimiento. Por defecto es 5.

    Retorno:
        motion: 'True' si se detecta movimiento, 'False' en caso contrario.
    """
    # Validación del parámetro 'ant'
    # if not ant:
    #     raise ValueError("'ant' no puede ser una lista vacía.")
    
    # Validación del parámetro 'act' 
    if not act:
        raise ValueError("'act' no puede ser una lista vacía.")

    # Validación de que 'thresh' sea un número positivo
    if not isinstance(thresh, (int, float)) or thresh <= 0:
        raise ValueError("El parámetro 'thresh' debe ser un número positivo.")
  
    # Ordenamiento de las listas de centroides para garantizar correspondencia entre elementos.
    ant.sort()
    act.sort()

    # Se asume que hay movimiento inicialmente.
    motion = True
    cont = 0    
  
    # Comparación de centroides correspondientes.
    for i in range(min(len(ant), len(act))):
        # Desempaqueto las coordenadas de los centroides.
        x1, y1 = ant[i]
        x2, y2 = act[i]

        # Verifico si el desplazamiento está por debajo del umbral en ambas coordenadas.
        if (x2 - x1) < thresh and (y2 - y1) < thresh:
            cont += 1
    
    # Si todos los centroides tienen desplazamientos inferiores al umbral, no hay movimiento.
    if cont == len(ant):  
        motion = False
    
    return motion

# ******************************************************************************************
# ******************************************************************************************