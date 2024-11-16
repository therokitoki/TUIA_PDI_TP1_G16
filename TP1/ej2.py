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
from customlib import imshow, letterAnswer, lineDetector, lineOrientation, questionROIDetector, headerDetector, letterBoxDetector, headerValidator
import warnings 
warnings.filterwarnings('ignore', category=FutureWarning) #Para evitar mensajes asociados a cambios planeados en futuras versiones de las librerías en uso

# ******************************************************************************************
# *                                     Implementación                                     *
# ******************************************************************************************

# Lectura de plantillas
bien_img = cv2.imread('.\\img\\BIEN.png')
mal_img = cv2.imread('.\\img\\MAL.png')
desaprobado_img = cv2.imread('.\\img\\DESAPROBADO.png')
desaprobado_img = cv2.cvtColor(desaprobado_img, cv2.COLOR_BGR2RGB)
aprobado_img = cv2.imread('.\\img\\APROBADO.png')
aprobado_img = cv2.cvtColor(aprobado_img, cv2.COLOR_BGR2RGB)
planilla_img = cv2.imread('.\\img\\PLANILLA.png')
planilla_img = cv2.cvtColor(planilla_img, cv2.COLOR_BGR2RGB)

desp_hori = 0 # Desplazamiento horizontal

# Se itera entre todos los exámenes.
for examen in range(1,6):

    # Lectura de imagen
    img = cv2.imread(f'.\\img\\examen_{examen}.png')

    # Se identifican las líneas de la imagen
    line_list = lineDetector(img, 200)
    img_lines_new = img.copy()

    # Se grafican las líneas encontradas
    for i in line_list:
        cv2.line(img_lines_new,i[0],i[1],(0,255,0),1)

    # Visualización de imagen - líneas encontradas
    #imshow(img_lines_new,title='Img con lineas post fix')

    # Se clasifican las líneas encontradas en horizontales y verticales
    h_lines, v_lines = lineOrientation(line_list)

    # Se identifican las regiones de interés (preguntas)
    questions = questionROIDetector(v_lines, h_lines, img, show= False)

    # En cada pregunta se identifica el segmento que contiene la respuesta y se la aisla.
    answers=[]
    for question in questions:
        letter_box = letterBoxDetector(img= question, show= False, header = False)
        answers.append(letterAnswer(letter_box[0]))

    print(f'\nExamen {examen}')
    print(f'Respuestas:')
    print(f'1:{answers[0]}, 2:{answers[1]}, 3:{answers[2]}, 4:{answers[3]}, 5:{answers[4]}, 6:{answers[5]}, 7:{answers[6]}, 8:{answers[7]}, 9:{answers[8]}, 10:{answers[9]}')

    # Se crea la lista que contiene las respuestas correctas
    correctos = ['C','B','A','D','B','B','A','B','D','D']
    correccion= []

    desp_vert = 0 # Desplazamiento vertical

    # Se compara la respuesta obtenida con la respuesta correcta
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

    print(f'\nCorrección respuestas:')
    print(f'Pregunta 1: {correccion[0]}\nPregunta 2: {correccion[1]}\nPregunta 3: {correccion[2]}\nPregunta 4: {correccion[3]}\nPregunta 5: {correccion[4]}\nPregunta 6: {correccion[5]}\nPregunta 7: {correccion[6]}\nPregunta 8: {correccion[7]}\nPregunta 9: {correccion[8]}\nPregunta 10: {correccion[9]}')

    # Se aisla el encabezado
    img_prb = headerDetector(h_lines= h_lines, show= False, img= img)

    # Extracción de datos del examen
    nombre, fecha, clase = letterBoxDetector(img_prb, False,header=True)
    nombre_rs = cv2.resize(nombre  , (275 , 35))

    # Se inserta el nombre del alumno en el reporte final
    planilla_img[242:242+ nombre_rs.shape[0], 12+desp_hori:12+desp_hori+nombre_rs.shape[1]] = nombre_rs
    
    # Se validan los datos del examen
    nombre_ok = headerValidator(nombre, 'name')
    fecha_ok = headerValidator(fecha, 'date')
    clase_ok = headerValidator(clase, 'class')
    header_check = [nombre_ok, fecha_ok, clase_ok]

    # Se ingresa la correción de los datos en el reporte final
    for header in header_check:
        if header:
            planilla_img[288+desp_vert:288+desp_vert + bien_img.shape[0], 195+desp_hori:195+desp_hori+bien_img.shape[1]] = bien_img
        else:
            planilla_img[288+desp_vert:288+desp_vert + mal_img.shape[0], 195+desp_hori:195+desp_hori+mal_img.shape[1]] = mal_img
        desp_vert += bien_img.shape[0] - 2
    print(f'Nombre: {nombre_ok}\nFecha: {fecha_ok}\nClase: {clase_ok}')

    if puntos_positivos >= 6:
        planilla_img[794:794+ aprobado_img.shape[0], 20+desp_hori:20+desp_hori+aprobado_img.shape[1]] = aprobado_img
    else:
        planilla_img[794:794+ desaprobado_img.shape[0], 20+desp_hori:20+desp_hori+desaprobado_img.shape[1]] = desaprobado_img

    desp_hori += 300

# Visualización del reporte final
imshow(planilla_img, colorbar= False)