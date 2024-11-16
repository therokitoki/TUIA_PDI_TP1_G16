############################################################################################
#                                                                                          #
#                              PROCESAMIENTO DE IMÁGENES 1                                 #
#                                 TRABAJO PRÁCTICO N°1                                     #
#                                                                                          #
#          GRUPO N°16: Gonzalo Asad, Sergio Castells, Agustín Alsop, Rocio Hachen          #
#                                                                                          #
#                       Problema 1 - Ecualización local de histograma                      #
#                                                                                          #
############################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_moneda = 'C:\\Users\\alsop\\Documents\\GitHub\\TUIA_PDI_TP1_G16\\TP2\\monedas.jpg'

#Cargo la imagen y la convierto a escala de grises
img = cv2.imread(img_moneda)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Aplico un filtro para reducir el ruido
#blurred = cv2.medianBlur(gray, 5)
blurred = cv2.GaussianBlur(gray, (11,11), 0)

#Detecto los bordes
edges = cv2.Canny(blurred, 30, 70)

#Aplico operaciones morfológicas

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
dilated = cv2.dilate(edges, kernel, iterations=1) # Dilatación para expandir las regiones detectadas
thresh_morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)  # Cierre para unir regiones cercanas

#Detecto contornos
contours, _ = cv2.findContours(thresh_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Creo una copia de la imagen original para rellenar los contornos
filled_image = img.copy()

#Rellenar los contornos detectados
for contour in contours:
    cv2.drawContours(filled_image, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)

#Visualización
plt.figure(figsize=(16, 8))

#Imagen original
ax1 = plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagen Original")
plt.axis("off")

#Detección de Bordes
plt.subplot(222, sharex = ax1, sharey = ax1)
plt.imshow(edges, cmap="gray")
plt.title("Umbralización Inicial")
plt.axis("off")

#Morfología
plt.subplot(223, sharex = ax1, sharey = ax1)
plt.imshow(thresh_morph, cmap="gray")
plt.title("Morfología")
plt.axis("off")

img_contornos = img.copy()
img_final = img.copy()
peso = 0
cent_50 = 0
cent_10 = 0
dados = []
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Ignorar áreas muy pequeñas
    if area < 100 or perimeter < 200:
        continue

    # Aproximar el contorno
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    # Calcular circularidad
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    # Coordenadas del contorno
    x, y, w, h = cv2.boundingRect(contour)
    # Identificar formas
    if len(approx) == 4:
        # Cuadrado
        # Recorta la región correspondiente al contorno
        cropped_region = thresh_morph[y:y+h, x:x+w]

        cv2.drawContours(img_contornos, [contour], -1, (0, 255, 255), thickness=cv2.FILLED)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18,18))
        cropped_region = cv2.morphologyEx(cropped_region, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60,60))
        cropped_region = cv2.morphologyEx(cropped_region, cv2.MORPH_OPEN, kernel_open)
        _, hierarchy = cv2.findContours(cropped_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        dados.append(len(hierarchy[0]))
        #Dibujar el rectángulo
        cv2.rectangle(img_final, (x, y), (x+w, y+h), (0, 255, 0), 3)

        #Agregar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_final, f'D: {len(hierarchy[0])}', (x+10, y-10), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    elif circularity > 0.2:
        # círculo
        if area < 80000:
            cv2.drawContours(img_contornos, [contour], -1, (100, 100, 0), thickness=cv2.FILLED)
            cent_10 += 1

            #Dibujar el rectángulo
            cv2.rectangle(img_final, (x, y), (x+w, y+h), (0, 255, 0), 3)

            #Agregar texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_final, 'M: $0,1', (x+10, y-10), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

        elif area >= 80000 and area < 100000:
            cv2.drawContours(img_contornos, [contour], -1, (50, 50, 0), thickness=cv2.FILLED)
            peso += 1

            #Dibujar el rectángulo
            cv2.rectangle(img_final, (x, y), (x+w, y+h), (0, 255, 0), 3)

            #Agregar texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_final, 'M: $1', (x+10, y-10), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
        else:
            cv2.drawContours(img_contornos, [contour], -1, (10, 10, 0), thickness=cv2.FILLED)
            cent_50 += 1

            #Dibujar el rectángulo
            cv2.rectangle(img_final, (x, y), (x+w, y+h), (0, 255, 0), 3)

            #Agregar texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_final, 'M: $0,5', (x+10, y-10), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
    else:
        # Otras formas
        cv2.drawContours(img_contornos, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)


#Contornos rellenados
plt.subplot(224, sharex = ax1, sharey = ax1)
plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
plt.title("Contornos Rellenados")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f'En total se detectaron {peso + cent_10 + cent_50} monedas')
print(f'De 1 peso: {peso}')
print(f'De 50 centavos: {cent_50}')
print(f'De 10 centavos: {cent_10}')
print(f'Además se detectaron {len(dados)} dados')
print(f'Los dados tienen en su cara superior los números {dados}')

plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
plt.title("Detección Automática de Objetos")
plt.show()
