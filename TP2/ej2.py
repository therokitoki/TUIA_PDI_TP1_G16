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

    img_final = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1- Umbralado
    _, thresh_img = cv2.threshold(gray, thresh=th, maxval=255, type=cv2.THRESH_BINARY)
    plt.imshow(thresh_img, cmap='gray')
    plt.title('Umbralado')
    plt.show()

    # 2- Componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, 8, cv2.CV_32S)
    plt.imshow(labels, cmap='gray')
    plt.title('Componentes conectadas')
    plt.show()

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


    # Imprimir labels filtrados
    mask_filtered_labels = np.isin(filtered_labels, filtered_index)

    filtered_labels[mask_filtered_labels] = 255
    filtered_labels[~mask_filtered_labels] = 0

    plt.imshow(filtered_labels, cmap='gray')
    plt.title('Filtrado por área')
    plt.show()


    # Imprimir letras filtradas
    mask_filtered_letters = np.isin(filtered_letters, letters_index)

    filtered_letters[mask_filtered_letters] = 255
    filtered_letters[~mask_filtered_letters] = 0

    plt.imshow(filtered_letters, cmap='gray')
    plt.title('Filtrado por relación de aspecto')
    plt.show()


    return cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)


img_auto = '.\img02.png'
img = cv2.imread(img_auto)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagen inicial')
plt.show()

# img_final = letterDetection(img, 150, 170.0, 30.0, 3.0, 1.5) # img01, img04, img05 (agarra un cosito más), img08 (agarra un cosito más)
img_final = letterDetection(img, 110, 170.0, 30.0, 3.0, 1.5) # img02

plt.imshow(img_final)
plt.title('Imagen final')
plt.show()