import cv2
import os
import numpy as np
import random
import shutil

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
originals_path = 'images/Imagenes_originales'
save_path = 'images/deteccion-bordes'

# Crear carpetas si no existen
os.makedirs(originals_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'Roberts'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'Prewitt'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'Sobel'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'Canny'), exist_ok=True)

# Obtener una lista de archivos y mezclar aleatoriamente
all_files = [f for f in os.listdir(base_path) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(all_files)

# Verificar cuántas imágenes ya hay en la carpeta de originales
existing_files = os.listdir(originals_path)
num_existing_files = len(existing_files)
num_needed_files = 10 - num_existing_files

# Seleccionar aleatoriamente las imágenes faltantes y copiarlas a la carpeta de originales
if num_needed_files > 0:
    selected_files = all_files[:num_needed_files]
    for filename in selected_files:
        original_file_path = os.path.join(base_path, filename)
        shutil.copy(original_file_path, os.path.join(originals_path, filename))
    existing_files.extend(selected_files)

# Procesar cada imagen en la carpeta de originales
for filename in existing_files:
    img_path = os.path.join(originals_path, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        print(f'Error al cargar la imagen: {filename}')
        continue

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

# Verificar las imágenes copiadas
copied_files = os.listdir(originals_path)
print(f'Imágenes copiadas: {copied_files}')



