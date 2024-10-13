import cv2
import numpy as np
import matplotlib.pyplot as plt

def local_hist_eq(img : np.ndarray, M : int, N : int) -> np.ndarray:
    
    """
    Implementa la ecualización local del histograma. 
    Parámetros:
        img: Imagen a procesar (en escala de grises)
        M: Ancho de la ventana de procesamiento (entero, positivo, impar)
        N: Alto de la ventana de procesamiento (entero, positivo, impar)
    """
    # Validación de parámetros
    if M < 0 or M//2 == 0 or type(M) != int:
        raise ValueError('El parámetro M debe ser entero, positivo e impar.')
    if N < 0 or N//2 == 0 or type(N) != int:
        raise ValueError('El parámetro N debe ser entero, positivo e impar.')

    # Validar si la imagen se encuentra en grayscale
    
    if len(img.shape) > 2:
        raise ValueError('La imagen proporcionada debe estar en escala de grises.')

    # Agregamos bordes para que todos los pixeles puedan ser analizados
    img_bordes = cv2.copyMakeBorder(img, M//2, M//2, N//2, N//2, cv2.BORDER_REPLICATE)
    img_salida = np.empty(img.shape)

    for i in range(img.shape[0]): #filas

        for j in range(img.shape[1]): #columnas
            
            ventana = img_bordes[i:i+M, j:j+N]

            hist_ventana = cv2.equalizeHist(ventana)

            img_salida[i, j] = hist_ventana[M//2, N//2] #reemplazo cada pixel por el pixel ecualizado
    
    return img_salida

# Visualizamos la imagen original
img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.show()

valores = [(3,3),(5,5),(11,11),(21,11),(35,25),(51,51),(101,101)]

plt.figure(figsize=(14,8))
ax = plt.subplot(2,4,1)
plt.imshow(img, cmap='gray')
plt.title('Img original')

for i in range(0,len(valores)):
    
    img_eq = local_hist_eq(img, valores[i][0], valores[i][1])
    plt.subplot(2, 4, i+2,sharex=ax,sharey=ax)
    plt.imshow(img_eq,cmap='gray')
    plt.title(f'Ventana {valores[i][0]}x{valores[i][1]}')
   
plt.show()


