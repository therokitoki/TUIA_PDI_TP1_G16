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
cap = cv2.VideoCapture('./videos/tirada_4.mp4')  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.
#print(fps)
frame_number = 0
while (cap.isOpened()): # Verifica si el video se abrió correctamente.

    ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

    if ret == True:

        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

    ###################################################################################################################################################
    # DETECTA PAÑO
    ###################################################################################################################################################    
        
        if frame_number == 0:
            frame_test = frame.copy()

            frame_test = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            L, A, B = cv2.split(frame_test)

            _, thresh_img = cv2.threshold(A, thresh=110, maxval=255, type=cv2.THRESH_BINARY_INV)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, 8, cv2.CV_32S)

            x,y,w,h,a = stats[1]

            x_ini = round(x+(w*0.05))
            y_ini = round(y+(h*0.05))
            x_fin = round(x+(w*0.95))
            y_fin = round(y+(h*0.95))

            #cv2.rectangle(frame, (x_ini, y_ini), (x_fin, y_fin), (0, 255, 0), 3)
            cv2.imwrite(os.path.join("./frames", f"frame_{frame_number}.jpg"), thresh_img) # Guarda el frame en el archivo './frames/frame_{frame_number}.jpg'.
        #video 1 == 65 perfeee
        #video 2 == 65 perfeee
        #video 3 == 65 perfeee
        #video 4 == 65 perfeee

    ###################################################################################################################################################
    ###################################################################################################################################################
    ###################################################################################################################################################    

        if (frame_number) > 30:
            frame_crop = frame[y_ini:y_fin, x_ini:x_fin]

            frame_crop_bgr = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2LAB)

            L, A, B = cv2.split(frame_crop_bgr)

            #_, thresh_img = cv2.threshold(A, thresh=110, maxval=255, type=cv2.THRESH_BINARY)

            #num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, 8, cv2.CV_32S)
            th_min = 95
            status = False
            max_area = 900
            min_area = 100
            jump = 1

    ###################################################################################################################################################
    # DETECTA DADOS
    ###################################################################################################################################################        

            while not status and th_min <= 120:
                _, thresh_img_a = cv2.threshold(A, thresh=th_min, maxval=255, type=cv2.THRESH_BINARY)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img_a, 8, cv2.CV_32S)

                counter_area = []
                cen_test = []

                if num_labels != 6:
                    th_min += jump
                    continue

                for i in range(len(stats)):
                    x, y, w, h, a = stats[i]

                    if a < max_area and a > min_area:
                        counter_area.append(i)
                        cen_test.append(centroids)

                # Si la cantidad de componentes identificadas es menor de 6, se descarta el umbral
                if len(counter_area) != 5:
                    th_min += jump
                    continue
                
                print(th_min)
                print(stats)
                status = True

    ###################################################################################################################################################
    ###################################################################################################################################################
    ###################################################################################################################################################

    ###################################################################################################################################################
    # DETECTA NUMEROS
    ###################################################################################################################################################            

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

                        cv2.imwrite(os.path.join("./frames", f"frame_finaaaaaaaaaaaaaaaaaal.jpg"), frame_crop)

                    #print(num_labels)

            #print(num_labels)
            #print(stats)
            #A para detectar dados
            #L para detectar numeros

            cv2.imwrite(os.path.join("./frames", f"frame_l.jpg"), L)
            cv2.imwrite(os.path.join("./frames", f"frame_frame_crop.jpg"), frame_crop)
            cv2.imwrite(os.path.join("./frames", f"frame_a.jpg"), A)
            cv2.imwrite(os.path.join("./frames", f"frame_b.jpg"), B)

    ###################################################################################################################################################
    ###################################################################################################################################################
    ###################################################################################################################################################        
        

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


# # --- Leer y grabar un video ------------------------------------------------
# cap =           cv2.VideoCapture('tirada_1.mp4')  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
# width =         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
# height =        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
# fps =           int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
# # n_frames =    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.

# out = cv2.VideoWriter('Video-Output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
# # Crear un objeto para escribir el video de salida.
# #   - 'Video-Output.mp4': Nombre del archivo de salida.
# #   - cv2.VideoWriter_fourcc(*'mp4v'): Codec utilizado para el archivo de salida.
# #   - fps: Cuadros por segundo del video de salida, debe coincidir con el video de entrada.
# #   - (width, height): Dimensiones del frame de salida, deben coincidir con las dimensiones originales del video.

# while (cap.isOpened()): # Verifica si el video se abrió correctamente.
    
#     ret, frame = cap.read()  # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

#     if ret == True:

#         cv2.rectangle(frame, (100,100), (200,200), (0,0,255), 2)

#         frame_show = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.

#         cv2.imshow('Frame', frame_show) # Muestra el frame redimensionado.

#         out.write(frame)   # Escribe el frame original (sin redimensionar) en el archivo de salida 'Video-Output.mp4'. IMPORTANTE: El tamaño del frame debe coincidir con el tamaño especificado al crear 'out'.
#         if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
#             break
#     else:
#         break

# cap.release() # Libera el objeto 'cap', cerrando el archivo.
# out.release() # Libera el objeto 'out',  cerrando el archivo.
# cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.