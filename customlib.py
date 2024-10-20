############################################################################################
#                                                                                          #
#                              PROCESAMIENTO DE IMÁGENES 1                                 #
#                                 TRABAJO PRÁCTICO N°1                                     #
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

def imshow(img : np.ndarray, new_fig : bool =True , title : str =None , color_img : bool =False , blocking : bool =True , colorbar : bool =True , ticks: bool =False ) -> None:
    """
    Función para visualizar imágenes

    Parámetros:
        img: Imagen a visualizar.
        new_fig: Si es Verdadero, se creará una nueva ventana de imagen.
        title: Título de la imagen, por default es None.
        color_img: Si es Verdadero, la imagen será considerada una imagen a color. De lo contrario, será considerada una imagen en escala de grises.
        blocking: Si es Verdadero, la ejecución del código se interrumpirá hasta que la ventana de la imagen sea cerrada.
        colorbar: Si es Verdadero, se visualizará la escala de colores a la derecha de la imagen.
        ticks: Si es Verdadero, xticks e yticks son deshabilitados.

    """
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)
    
    # ******************************************************************************************

def letterAnswer(letter_box: np.ndarray) -> str:

    """
    Identifica una letra a partir de una imagen.

    Parámetros:
        letter_box: imagen a procesar

    Retorno:
        str: Letra identificada
    """

    # Transformación de la imagen a grayscale
    letter_box = cv2.cvtColor(letter_box, cv2.COLOR_BGR2GRAY)

    # Cantidad de píxeles oscuros
    pixeles_debajo_150 = letter_box < 150
    cantidad_pixeles = np.sum(pixeles_debajo_150) 

    # Umbralado
    _, thresh_img = cv2.threshold(letter_box, thresh=230, maxval=255, type=cv2.THRESH_BINARY)

    # Validación: Se verifica si el alumno ingresó más de una letra o dejó la casilla en blanco
    if cantidad_pixeles > 40:
        letter= 'INVÁLIDO'
    elif cantidad_pixeles == 0:
        letter= 'NO RESPONDE'
    else:
        # Contornos de la imagen
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(letter_box, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)

        # Dependiendo de la cantidad de contornos, se clasifica entre B (4 contornos), C (2 contornos), A/D (3 contornos)
        if len(hierarchy[0])==4:
            letter= 'B'
        elif len(hierarchy[0])==2:
            letter= 'C'
        elif len(hierarchy[0])==3:
            # Para diferenciar entre A y D, se analizan las áreas de los contornos
            connectivity = 8
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, connectivity, cv2.CV_32S)
            if stats[2][4]<15:
                letter= 'A'
            else:
                letter= 'D'
        else:
            letter= 'INVÁLIDO'
    
    # Se retorna la letra identificada
    return(letter)

 # ******************************************************************************************

def lineDetector(src : np.ndarray, th : int) -> list[list[tuple]]:

    """

    Identifica líneas rectas en una imagen.

    Parámetros:
        src: Imagen a procesar.
        th: Umbral a utilizar en la función HoughLines.

    Retorno:
        list: Lista de listas de coordenadas de las líneas identificadas.

    """

    # Transformación de la imagen a grayscale
    gray = cv2.cvtColor(src, cv2.IMREAD_GRAYSCALE)   
    
    # Visualización de imagen
    #imshow(gray,title='Imagen original grayscale')

    # Identificación de bordes mediante Canny
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)

    # Detección de líneas rectas con HoughLines
    src_lines = src.copy()
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=th)

    if lines is None:
        th = th - 10
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=th)

    # Extracción de coordenadas de líneas
    line_list = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a=np.cos(theta)
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho
        x1=int(x0+1000*(-b))
        y1=int(y0+1000*(a))
        x2=int(x0-1000*(-b))
        y2=int(y0-1000*(a))
        line_list.append([(x1,y1),(x2,y2)])
        cv2.line(src_lines,(x1,y1),(x2,y2),(0,255,0),1)

    # Visualización de imagen
    #imshow(src_lines,title='Imagen con lineas pre-fix')

    # Se promedian las líneas cercanas entre sí para obtener una única línea entre ellas
    final_line_list = []
    for i in line_list:
        for j in line_list:
            dif_x = abs(i[0][0] - j[0][0]) + abs(i[1][0] - j[1][0])
            dif_y = abs(i[0][1] - j[0][1]) + abs(i[1][1] - j[1][1])
            if dif_x == 0 and dif_y < 10 and dif_y != 0:
                new_line = [(-1000, int((i[0][1] + j[0][1])/2)), (1000, int((i[1][1] + j[1][1])/2))]
                if new_line not in final_line_list:
                    final_line_list.append(new_line)
            if dif_x < 10 and dif_x != 0 and dif_y == 0:
                new_line = [(int((i[0][0] + j[0][0])/2), 1000), (int((i[1][0] + j[1][0])/2), -1000)]
                if new_line not in final_line_list:
                    final_line_list.append(new_line)
    
    # Se retorna la lista de líneas
    return final_line_list

# ******************************************************************************************

def lineOrientation(line_list : list[list[tuple]]) -> tuple:

    """
    Clasifica las líneas entre horizontales y verticales

    Parámetros:
        line_list: Lista de listas de coordenadas de las líneas identificadas.

    Retorno:
        tuple:  El primer elemento es una lista de listas de coordenadas de las líneas horizontales
                El segundo elemento es una lista de listas de coordenadas de las líneas verticales
    """

    # Clasifico las líneas entre horizontales y verticales
    h_lines = []
    v_lines = []
    for i in line_list:
        if i[0][0] == -1000:
            h_lines.append(i)
            print(f'Linea Horizontal {i[0][0]}')
        elif i[0][1] == 1000:
            v_lines.append(i)
            print(f'Linea Vertical {i[0][1]}')
        else: # línea oblicua
            pass

    # Las ordeno (me será útil más adelante)
    h_lines.sort(key = lambda x: x[1])
    v_lines.sort(key = lambda x: x[0])
    
    return h_lines, v_lines

# ******************************************************************************************

def questionROIDetector(v_lines : list, h_lines : list, img : np.ndarray, show : bool = True) -> list[np.ndarray] :

    """
    Utiliza las intersecciones entre líneas verticales y horizontales para identificar los ROIs.

    Parámetros:
        v_lines: Lista de listas de coordenadas de líneas verticales.
        h_lines: Lista de listas de coordenadas de líneas horizontales.
        img: Imagen a partir de la cual se obtendrán los ROIs
        show: Si es Verdadero, se visualizarán los ROIs obtenidos.

    Retorno:
        list[np.ndarray]: Lista de ROIs obtenidos

    """

    # Inicializamos la lista de ROIs
    roi_list = []

    idx_v = 0
    for j in range(len(v_lines)-2):

        x1 = int(v_lines[idx_v][0][0])
        x2 = int(v_lines[idx_v + 1][0][0])
        idx_v += 2

        idx_h = 1

        for i in range(1, len(h_lines) - 1):

            # Recortamos la celda de la imagen en escala de grises
            y1 = int(h_lines[idx_h][0][1])
            y2 = int(h_lines[idx_h + 1][0][1])
            idx_h += 1

            roi = img[y1+4:y2-4, x1+4:x2-4]

            # Visualización de imagen
            if show:
                imshow(roi)

            roi_list.append(roi)

    return (roi_list)

# ******************************************************************************************

def headerDetector(h_lines: list, img: np.ndarray, show: bool = True) -> np.ndarray:
    """

    Identifica el encabezado de una imagen en base a las líneas horizontales.

    Parámetros:
        h_lines: Lista de listas de coordenadas de líneas horizontales.
        img: Imagen a partir de la cual se obtendrá el encabezado
        show: Si es Verdadero, se visualizará el encabezado obtenido.

    Retorno:
        np.ndarray: Encabezado obtenido.

    """

    # Se define la región del encabezado
    x1 = 0
    x2 = img.shape[1]
    y1 = 0
    y2 = int(h_lines[0][0][1]) + 10

    roi = img[y1+4:y2-4, x1+4:x2-4]

    # Visualización de imagen
    if show:
        imshow(roi)

    return roi

# ******************************************************************************************

def letterBoxDetector(img: np.ndarray, show: bool = True, header: bool = False) -> list:

    """
    
    Dado una imagen, identifica el recuadro en el cual se encuentra una letra.

    Parámetros:
        img: Imagen a procesar.
        show: Si es Verdadero, se visualizará el recuadro obtenido.
        header: Si es Verdadero, analizará una sección especifica de la imagen.

    Retorno:
        list: Lista de recuadros obtenidos.

    """

    letter_box = []

    # Transformación de la imagen a grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Umbralado
    umbral, thresh_img = cv2.threshold(gray, 150, 255, type=cv2.THRESH_BINARY_INV)

    # Visualización de imagen
    #imshow(thresh_img)

    # Componentes conectadas
    num_labels, labels_im = cv2.connectedComponents(thresh_img)

    # Dibuja las componentes conectadas
    output = np.zeros(img.shape, dtype=np.uint8)

    # Creación de una máscara para dibujar solo las componentes que podrían ser líneas
    for label in range(1, num_labels):

        component_mask = (labels_im == label).astype("uint8") * 255
        x, y, w, h = cv2.boundingRect(component_mask)

        # Filtrado por el tamaño de las componentes, asumiendo que la línea es la más larga y delgada
        aspect_ratio = w / h
        if aspect_ratio > 5:  # Línea larga y delgada
            if header:
                test = img.copy()
                z = y - 20
                roi = test[z:z+16, x:x+w,]
            else:
                test = img.copy()
                z = y-14
                roi = test[z:z+14, x:x+w,]
            letter_box.append(roi)

            # Visualización de imagen
            if show:
                imshow(roi)

    return letter_box

# ******************************************************************************************

def headerValidator(img: np.ndarray, field: str = 'name') -> bool:

    """
    Analiza morfología de distintos sectores del encabezado.

    Parámetros:
        img: Encabezado a procesar.
        field: Sector a analizar (name, date, class)

    Retorno:
        bool: Resultado del análisis (True: El sector es válido, False: El sector no es válido)

    """
    # Transformación de la imagen a grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Umbralado
    umbral, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Implementación de connectedComponentsWithStats para contar componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, connectivity=8)

    field = field.lower()

    # Validación de parámetro field
    if field not in ['name', 'date', 'class']:
        raise ValueError("El parámetro field debe ser 'name', 'date' o 'class'")

    if field == 'name':

        # Se verifica que el número de componentes conectadas sea al menos 3 (2 correspondientes al nombre y apellido y una por el fondo de la imagen)
        if num_labels >= 3:

        # Se verifica la existencia de un espacio entre "Nombre" y "Apellido"
            x_ant = stats[1][0]
            for i in range(2, num_labels):
                x, y, w, h, a = stats[i]
                if x - x_ant > 12:
                    return True
                x_ant = x
            return False

    elif field == 'date':
        # Se verifica que el número de componentes conectadas sea 9 (8 correspondientes a la fecha y una por el fondo de la imagen)
        if num_labels == 9:
            return True
        return False

    else:
        # Se verifica que el número de componentes conectadas sea 2 (1 correspondiente a la clase y una por el fondo de la imagen)
        if num_labels == 2:
            return True
        return False
