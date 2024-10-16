from cProfile import label
import cv2
import numpy as np
import matplotlib.pyplot as plt

def answer_line(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)  
    pixel_negros = np.sum(thresh == 0, axis=1)  
    #print(pixel_negros)
    indice = np.argmax(pixel_negros) #row con mayoria de pixeles negros
    cv2.line(roi, (0, indice), (roi.shape[1], indice), (255, 0, 0), 1)  
    imshow(roi)
    return indice

def opcion2(img):
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
            test = img.copy()
            z = y-14
            crop = test[z:z+14, x:x+w,]
            # b = 33
            # d = 29
            # a = 28
            # c = 22
            gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            pixeles_debajo_150 = gray2 < 150
            cantidad_pixeles = np.sum(pixeles_debajo_150)
            umbral, thresh_img2 = cv2.threshold(gray2, thresh=230, maxval=255, type=cv2.THRESH_BINARY)
            if cantidad_pixeles > 40:
                valores= 'INVÁLIDO'
            elif cantidad_pixeles == 0:
                valores= 'NO RESPONDE'
            else:
                contours, hierarchy = cv2.findContours(thresh_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(crop, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)  # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc

                if len(hierarchy[0])==4:
                    valores= 'B'
                elif len(hierarchy[0])==2:
                    valores= 'C'
                else:
                    print("Entreeeee")
                    connectivity = 8
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img2, connectivity, cv2.CV_32S)
                    if stats[2][4]<15:
                        valores= 'A'
                    else:
                        valores= 'D'
            cv2.rectangle(img, (x, y), (x+w, y-14), (0, 255, 0), 1)
            #imshow(crop)
            output = cv2.bitwise_or(output, cv2.merge([component_mask, component_mask, component_mask]))

    # Mostrar la imagen con la línea detectada
    #imshow(img)
    return(valores)

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=True, colorbar=True, ticks=False):
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


def line_detector(src : np.ndarray, th : int, for_roi = False) -> list[list[tuple]]:
    gray = cv2.cvtColor(src, cv2.IMREAD_GRAYSCALE)   # Transformamos la imagen a grayscale
    if not for_roi:
        imshow(gray,title='Img original gray')

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

    if not for_roi:
        imshow(src_lines,title='Imagen con lineas pre fix')

    # El problema es que me detectó 40 líneas en lugar de 20 (por la parte superior y la inferior)
    # Esto me puede ocasionar problemas a la hora de detectar los cuadrados

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


# Leemos la imagen a color y la pasamos a esacala de grises
img = cv2.imread('examen_5.png')
#img2 = cv2.imread('examen_3.png', cv2.IMREAD_GRAYSCALE)   # Leemos imagen

# Graficamos las nuevas líneas
line_list = line_detector(img, 200)
img_lines_new = img.copy()

for i in line_list:
    cv2.line(img_lines_new,i[0],i[1],(0,255,0),1)

imshow(img_lines_new,title='Img con lineas post fix')

h_lines, v_lines = line_orientation(line_list)

questions = question_roi_detector(v_lines, h_lines, img, show= False)
valores=[]
for question in questions:
    valores.append(opcion2(question))
print(f'Las respuestas fueron 1:{valores[0]} 2:{valores[1]} 3:{valores[2]} 4:{valores[3]} 5:{valores[4]} 6:{valores[5]} 7:{valores[6]} 8:{valores[7]} 9:{valores[8]} 10:{valores[9]}')
correctos = ['C','B','A','D','B','B','A','B','D','D']
correccion= []
for i in range(0,10):
    if valores[i] == correctos[i]:
        correccion.append('OK')
    else:
        correccion.append('MAL')

print(f'Pregunta 1: {correccion[0]}\nPregunta 2: {correccion[1]}\nPregunta 3: {correccion[2]}\nPregunta 4: {correccion[3]}\nPregunta 5: {correccion[4]}\nPregunta 6: {correccion[5]}\nPregunta 7: {correccion[6]}\nPregunta 8: {correccion[7]}\nPregunta 9: {correccion[8]}\nPregunta 10: {correccion[9]}')