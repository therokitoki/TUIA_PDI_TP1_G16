############################################################################################
#                                                                                          #
#                              PROCESAMIENTO DE IMÁGENES 1                                 #
#                                 TRABAJO PRÁCTICO N°3                                     #
#                                                                                          #
#          GRUPO N°16: Gonzalo Asad, Sergio Castells, Agustín Alsop, Rocio Hachen          #
#                                                                                          #
#                               Problema 1 - Cinco dados                                   #
#                                                                                          #
############################################################################################

# Librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import false
from customlib import *
import os

os.makedirs("./frames", exist_ok = True)  # Si no existe, crea la carpeta 'frames' en el directorio actual.

# --- Leer un video --------------------------------------------
cap = cv2.VideoCapture('./videos/tirada_2.mp4')  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.
#print(fps)
frame_number = 0
centroids_ant = [(0,0),(0,0),(0,0),(0,0),(0,0)]
while (cap.isOpened()): # Verifica si el video se abrió correctamente.

    ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

    if ret == True:

        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.
        
        # Detección de Región de Interés (paño)
        if frame_number == 0:
            x_ini, x_fin, y_ini, y_fin = roiDetect(img=frame, percent=5, thresh=110, save=True)


        if (frame_number) > 0: # Ver de Eliminar esto
            frame_crop = frame[y_ini:y_fin, x_ini:x_fin]

            frame_crop_bgr = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2LAB)

            L, A, B = cv2.split(frame_crop_bgr)

            flag, centroids, stats = centroidsDetect(img=A, th_min=95, min_area=100, max_area=900, jump=1)
  
            # Comparación de Centroides
            if flag:   # Se detectaron los 5 dados
                pass
             
                 
            ######

            for stat in stats:
                x,y,w,h,a = stat
                if a < max_area and a > min_area:
                    if a < ((x_fin-x_ini)*(y_fin-y_ini)*0.95):
                        cv2.rectangle(frame_crop, (x, y), (x+w, y+h), (255, 0, 0), 1)

                        frame_l_crop = L[y:y+h, x:x+w]

                        _, thresh_img_l = cv2.threshold(frame_l_crop, thresh=195, maxval=255, type=cv2.THRESH_BINARY)

                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img_l, 8, cv2.CV_32S)

                        font = cv2.FONT_HERSHEY_SIMPLEX

                        cv2.putText(frame_crop, f'N {num_labels-1}', (x, y-5), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                        


        cv2.imshow('Frame', frame) # Muestra el frame redimensionado.

        # cv2.imwrite(os.path.join("frames", f"frame_{frame_number}.jpg"), frame) # Guarda el frame en el archivo './frames/frame_{frame_number}.jpg'.

        frame_number += 1
        if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
            break
    else:  
        break  
#print(frame_number)

cap.release() # Libera el objeto 'cap', cerrando el archivo.
cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.