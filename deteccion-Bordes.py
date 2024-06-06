import cv2
import os
import numpy as np
import random

# Funciones de detección de bordes
def roberts_cross(image):
    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, 1], [-1, 0]], dtype=int)
    roberts_x = cv2.filter2D(image, cv2.CV_16S, kernelx)
    roberts_y = cv2.filter2D(image, cv2.CV_16S, kernely)
    roberts = cv2.convertScaleAbs(roberts_x) + cv2.convertScaleAbs(roberts_y)
    return roberts

def prewitt(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    prewitt_x = cv2.filter2D(image, -1, kernelx)
    prewitt_y = cv2.filter2D(image, -1, kernely)
    prewitt = cv2.convertScaleAbs(prewitt_x) + cv2.convertScaleAbs(prewitt_y)
    return prewitt

def sobel(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobelx) + cv2.convertScaleAbs(sobely)
    return sobel

def canny(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Ruta de las carpetas
base_path = 'images/base/BoneFractureDataset/testing/fractured'
save_path = 'images/deteccion-bordes'

# Crear carpetas si no existen
os.makedirs(os.path.join(save_path, 'Roberts'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'Prewitt'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'Sobel'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'Canny'), exist_ok=True)

# Obtener una lista de archivos y mezclar aleatoriamente
all_files = [f for f in os.listdir(base_path) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(all_files)

# Contador de imágenes procesadas
count = 0
max_images = 10

# Procesar cada imagen en la carpeta base de forma aleatoria
for filename in all_files:
    if count >= max_images:
        break
    img_path = os.path.join(base_path, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Aplicar algoritmos
    edges_roberts = roberts_cross(img)
    edges_prewitt = prewitt(img)
    edges_sobel = sobel(img)
    edges_canny = canny(img)

    # Guardar resultados
    cv2.imwrite(os.path.join(save_path, 'Roberts', f'roberts_{filename}'), edges_roberts)
    cv2.imwrite(os.path.join(save_path, 'Prewitt', f'prewitt_{filename}'), edges_prewitt)
    cv2.imwrite(os.path.join(save_path, 'Sobel', f'sobel_{filename}'), edges_sobel)
    cv2.imwrite(os.path.join(save_path, 'Canny', f'canny_{filename}'), edges_canny)

    # Incrementar el contador
    count += 1

