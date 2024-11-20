############################################################################################
#                                                                                          #
#                              PROCESAMIENTO DE IMÁGENES 1                                 #
#                                 TRABAJO PRÁCTICO N°2                                     #
#                                                                                          #
#          GRUPO N°16: Gonzalo Asad, Sergio Castells, Agustín Alsop, Rocio Hachen          #
#                                                                                          #
#                           Problema 2 - Detección de patentes                             #
#                                                                                          #
############################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

def letterDetection(img: np.ndarray, th: int, max_area: float, min_area: float, max_aspect_ratio: float, min_aspect_ratio: float) -> np.ndarray:


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #gray_eq = cv2.equalizeHist(gray)
    
    # plt.imshow(gray_eq, cmap='gray')
    # plt.title('Imagen EQ')
    # plt.show()
    
    img_valida = False
    while not img_valida and th <= 250:
        
        img_final = img.copy()
        # 1- Umbralado
        _, thresh_img = cv2.threshold(gray, thresh=th, maxval=255, type=cv2.THRESH_BINARY)
        # plt.imshow(thresh_img, cmap='gray')
        # plt.title('Umbralado')
        # plt.show()

        # 2- Componentes conectadas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, 8, cv2.CV_32S)
        # plt.imshow(labels, cmap='gray')
        # plt.title('Componentes conectadas')
        # plt.show()

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
            th += 1
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

        #print(maximo_x, 'maximo')

        if maximo_x > 25:
            th += 1
            continue

        for y in coord_y:
            dist = y-y_ant
            if maximo_y < dist:
                maximo_y = dist
            y_ant = y

        if maximo_y > 5:
            th += 1
            continue

        cv2.rectangle(img_final, (coord_x[0]-5, coord_y[0]-round((alturas[5]//2)*1.5)), (coord_x[5]+round((coord_x[5]-coord_x[4])*1.5), coord_y[0]+(alturas[5]*2)), (0, 255, 0), 1)
        # Imprimir labels filtrados
        mask_filtered_labels = np.isin(filtered_labels, filtered_index)

        filtered_labels[mask_filtered_labels] = 255
        filtered_labels[~mask_filtered_labels] = 0

        # plt.imshow(filtered_labels, cmap='gray')
        # plt.title('Filtrado por área')
        # plt.show()


        # Imprimir letras filtradas
        mask_filtered_letters = np.isin(filtered_letters, letters_index)

        filtered_letters[mask_filtered_letters] = 255
        filtered_letters[~mask_filtered_letters] = 0

        # plt.imshow(filtered_letters, cmap='gray')
        # plt.title('Filtrado por relación de aspecto')
        # plt.show()
        img_valida = True
        #print(th)

    print(f"Umbral final: {th}")
    return cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)

for i in range(1,13):
    if i < 10:
        img_auto = f'.\img0{i}.png'
    else:
        img_auto = f'.\img{i}.png'

    img = cv2.imread(img_auto)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Imagen inicial')
    plt.show()

    img_final = letterDetection(img, 1, 170.0, 30.0, 3.0, 1.5)
    # umbral img 02: 110
    plt.imshow(img_final)
    plt.title('Imagen final')
    plt.show()