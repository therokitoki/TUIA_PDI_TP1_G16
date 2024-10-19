############################################################################################
#                                                                                          #
#                              PROCESAMIENTO DE IMÁGENES 1                                 #
#                                 TRABAJO PRÁCTICO N°1                                     #
#                                                                                          #
#          GRUPO N°16: Gonzalo Asad, Sergio Castells, Agustín Alsop, Rocio Hachen          #                                                                                          
#                                                                                          #
#                       Problema 2 - Corrección de múltiple choice                         #
#                                                                                          #
############################################################################################

from cProfile import label
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ******************************************************************************************
# *                               Declaración de Funciones                                 *
# ******************************************************************************************
def imshow(img, new_fig=True, title=None, color_img=False, blocking=True, colorbar=True, ticks=False):
    """
    Función para visualizar imágenes

    Parámetros:
        img: Imagen a procesar (en escala de grises)
        new_: Ancho de la ventana de procesamiento (entero, positivo, impar)
        height: Alto de la ventana de procesamiento (entero, positivo, impar)

    Retorno:
        str: Letra identificada
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

def letterAnswer(letter_box: np.ndarray) -> str:

    """
    Identifica una letra a partir de una imagen.

    Parámetros:
        letter_box: imagen a procesar

    Retorno:
        str: Letra identificada
    """

    letter_box = cv2.cvtColor(letter_box, cv2.COLOR_BGR2GRAY)
    pixeles_debajo_150 = letter_box < 150
    cantidad_pixeles = np.sum(pixeles_debajo_150)
    _, thresh_img = cv2.threshold(letter_box, thresh=230, maxval=255, type=cv2.THRESH_BINARY)
    if cantidad_pixeles > 40:
        letter= 'INVÁLIDO'
    elif cantidad_pixeles == 0:
        letter= 'NO RESPONDE'
    else:
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(letter_box, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)
        if len(hierarchy[0])==4:
            letter= 'B'
        elif len(hierarchy[0])==2:
            letter= 'C'
        else:
            connectivity = 8
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, connectivity, cv2.CV_32S)
            if stats[2][4]<15:
                letter= 'A'
            else:
                letter= 'D'
    return(letter)


def line_detector(src : np.ndarray, th : int, for_roi = False) -> list[list[tuple]]:
    gray = cv2.cvtColor(src, cv2.IMREAD_GRAYSCALE)   # Transformamos la imagen a grayscale
    #if not for_roi:
       # imshow(gray,title='Img original gray')

    #Obtenemos los bordes mediante Canny
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)

    # Creo las líneas rectas con HoughLines
    src_lines = src.copy()
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=th)   # https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a

    if lines is None:
        th = th - 10
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=th)
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

    #if not for_roi:
        #imshow(src_lines,title='Imagen con lineas pre fix')

    # Buscamos líneas cercanas entre sí y creamos las líneas 'promedio'
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
    return final_line_list


def line_orientation(line_list : list[list[tuple]]) -> list[list[tuple]]:
    # Ahora debería:
    #   - Delimitar los cuadrados con las intersecciones de las líneas
    #   - Ver si la cantidad de píxeles negros es menor a cierto umbral
    #   - Si la condición anterior se cumple, sumarlo como casillero vacío y rellenarlo de gris

    # Clasifico las líneas entre horizontales y verticales
    h_lines = []
    v_lines = []
    for i in line_list:
        if i[0][0] == -1000:
            h_lines.append(i)
        else:
            v_lines.append(i)

    # Las ordeno (me será útil más adelante)
    h_lines.sort(key = lambda x: x[1])
    v_lines.sort(key = lambda x: x[0])

    return h_lines, v_lines


def question_roi_detector(v_lines : list, h_lines : list, img : np.ndarray, show = True) -> list[np.ndarray]:
    # Creamos los roi
    roi_list = []
    idx_v = 0
    for j in range(len(v_lines)-2):

        x1 = int(v_lines[idx_v][0][0])
        x2 = int(v_lines[idx_v + 1][0][0])
        idx_v += 2

        idx_h = 1

        for i in range(1, len(h_lines) - 1):
            # Recorto la celda de la imagen en escala de grises
            y1 = int(h_lines[idx_h][0][1])
            y2 = int(h_lines[idx_h + 1][0][1])
            idx_h += 1

            roi = img[y1+4:y2-4, x1+4:x2-4]
            if show:
                imshow(roi)
            roi_list.append(roi)
    return (roi_list)

##################################################################################################################################################################################
########################################################################### TPT ESTUVO AQUÍ ######################################################################################
##################################################################################################################################################################################

# *******************************************************
# *                 Defino Funciones                    *
# *******************************************************

def headerDetector(h_lines: list, img: np.ndarray, show: bool = True) -> np.ndarray:

    x1 = 0
    x2 = img.shape[1]
    y1 = 0
    y2 = int(h_lines[0][0][1]) + 10

    roi = img[y1+4:y2-4, x1+4:x2-4]

    if show:
        imshow(roi)

    return roi


def letterBoxDetector(img: np.ndarray, show: bool = True, header: bool = False) -> list:
    letter_box = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    umbral, thresh_img = cv2.threshold(gray, 150, 255, type=cv2.THRESH_BINARY_INV)
    #imshow(thresh_img)
    num_labels, labels_im = cv2.connectedComponents(thresh_img)

    # Dibuja las componentes conectadas
    output = np.zeros(img.shape, dtype=np.uint8)
    # Crear una máscara para dibujar solo las componentes que podrían ser líneas
    for label in range(1, num_labels):

        component_mask = (labels_im == label).astype("uint8") * 255
        x, y, w, h = cv2.boundingRect(component_mask)

        # Filtrar por el tamaño de las componentes, asumiendo que la línea es la más larga y delgada
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
            if show:
                imshow(roi)

    return letter_box


def headerValidator(img: np.ndarray, field: str = 'name') -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    umbral, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Aplico connectedComponentsWithStats para contar componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, connectivity=8)

    field = field.lower()

    # Valido parámetro field
    if field not in ['name', 'date', 'class']:
        raise ValueError("El parámetro field debe ser 'name', 'date' o 'class'")

    if field == 'name':
        # Verifico que el número de componentes conectadas sea al menos 3 (2 correspondientes al nombre y apellido y una por el fondo de la imagen)
        if num_labels >= 3:
        # Verifico la existencia de un espacio entre "Nombre" y "Apellido"
            x_ant = stats[1][0]
            for i in range(2, num_labels):
                x, y, w, h, a = stats[i]
                if x - x_ant > 12:
                    return True
                x_ant = x
            return False

    elif field == 'date':
        # Verifico que el número de componentes conectadas sea 8 (8 correspondientes a la fecha y una por el fondo de la imagen)
        if num_labels == 9:
            return True
        return False

    else:
        # Verifico que el número de componentes conectadas sea 2 (1 correspondiente a la clase y una por el fondo de la imagen)
        if num_labels == 2:
            return True
        return False

# *******************************************************
# *                    Implemento                       *
# *******************************************************
bien_img = cv2.imread('BIEN.png')
mal_img = cv2.imread('MAL.png')
desaprobado_img = cv2.imread('DESAPROBADO.png')
aprobado_img = cv2.imread('APROBADO.png')
planilla_img = cv2.imread('PLANILLA.png')

# Leemos la imagen a color y la pasamos a esacala de grises
desp_hori = 0
for examen in range(1,6):

    img = cv2.imread(f'examen_{examen}.png')

    # Graficamos las nuevas líneas
    line_list = line_detector(img, 200)
    img_lines_new = img.copy()

    for i in line_list:
        cv2.line(img_lines_new,i[0],i[1],(0,255,0),1)

    #imshow(img_lines_new,title='Img con lineas post fix')

    h_lines, v_lines = line_orientation(line_list)

    questions = question_roi_detector(v_lines, h_lines, img, show= False)
    answers=[]
    for question in questions:
        letter_box = letterBoxDetector(img= question, show= False, header = False)
        answers.append(letterAnswer(letter_box[0]))
    print(f'Las respuestas fueron 1:{answers[0]} 2:{answers[1]} 3:{answers[2]} 4:{answers[3]} 5:{answers[4]} 6:{answers[5]} 7:{answers[6]} 8:{answers[7]} 9:{answers[8]} 10:{answers[9]}')
    correctos = ['C','B','A','D','B','B','A','B','D','D']
    correccion= []
    desp_vert = 0
    puntos_positivos= 0
    for i in range(0,10):
        if answers[i] == correctos[i]:
            puntos_positivos += 1
            correccion.append('OK')
            planilla_img[288+desp_vert:288+desp_vert + bien_img.shape[0], 195+desp_hori:195+desp_hori+bien_img.shape[1]] = bien_img
        else:
            correccion.append('MAL')
            planilla_img[288+desp_vert:288+desp_vert + mal_img.shape[0], 195+desp_hori:195+desp_hori+mal_img.shape[1]] = mal_img
        desp_vert += bien_img.shape[0] - 2

    print(f'Pregunta 1: {correccion[0]}\nPregunta 2: {correccion[1]}\nPregunta 3: {correccion[2]}\nPregunta 4: {correccion[3]}\nPregunta 5: {correccion[4]}\nPregunta 6: {correccion[5]}\nPregunta 7: {correccion[6]}\nPregunta 8: {correccion[7]}\nPregunta 9: {correccion[8]}\nPregunta 10: {correccion[9]}')

    img_prb = headerDetector(h_lines= h_lines, show= False, img= img)

    nombre, fecha, clase = letterBoxDetector(img_prb, False,header=True)
    nombre_rs = cv2.resize(nombre  , (275 , 35))
    planilla_img[245:245+ nombre_rs.shape[0], 10+desp_hori:10+desp_hori+nombre_rs.shape[1]] = nombre_rs

    nombre_ok = headerValidator(nombre, 'name')
    fecha_ok = headerValidator(fecha, 'date')
    clase_ok = headerValidator(clase, 'class')
    header_check = [nombre_ok, fecha_ok, clase_ok]
    for header in header_check:
        if header:
            planilla_img[288+desp_vert:288+desp_vert + bien_img.shape[0], 195+desp_hori:195+desp_hori+bien_img.shape[1]] = bien_img
        else:
            planilla_img[288+desp_vert:288+desp_vert + mal_img.shape[0], 195+desp_hori:195+desp_hori+mal_img.shape[1]] = mal_img
        desp_vert += bien_img.shape[0] - 2
    print(f'Nombre: {nombre_ok}\nFecha: {fecha_ok}\nClase: {clase_ok}')

    if puntos_positivos >= 6:
        planilla_img[790:790+ aprobado_img.shape[0], 20+desp_hori:20+desp_hori+aprobado_img.shape[1]] = aprobado_img
    else:
        planilla_img[790:790+ desaprobado_img.shape[0], 20+desp_hori:20+desp_hori+desaprobado_img.shape[1]] = desaprobado_img

    desp_hori += 300

imshow(planilla_img)