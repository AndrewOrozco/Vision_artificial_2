import cv2
import os
import numpy as np
import random
import shutil

# Funciones de filtros
def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def high_pass_filter(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Ruta de las carpetas
originals_path = 'images/Imagenes_originales'
save_path = 'images/filtros-espaciales'

# Crear carpetas si no existen
os.makedirs(os.path.join(save_path, 'paso-bajo'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'paso-alto'), exist_ok=True)

# Obtener una lista de archivos en la carpeta de originales
all_files = [f for f in os.listdir(originals_path) if f.endswith('.jpg') or f.endswith('.png')]

# Procesar cada imagen en la carpeta de originales
for filename in all_files:
    img_path = os.path.join(originals_path, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        print(f'Error al cargar la imagen: {filename}')
        continue

    # Aplicar filtros
    gaussian = gaussian_blur(img)
    high_pass = high_pass_filter(img)

    # Guardar resultados
    cv2.imwrite(os.path.join(save_path, 'paso-bajo', f'pasobajo_{filename}'), gaussian)
    cv2.imwrite(os.path.join(save_path, 'paso-alto', f'pasoalto_{filename}'), high_pass)

# Verificar las imágenes procesadas
processed_files_paso_bajo = os.listdir(os.path.join(save_path, 'paso-bajo'))
processed_files_paso_alto = os.listdir(os.path.join(save_path, 'paso-alto'))
print(f'Imágenes procesadas (Paso Bajo): {processed_files_paso_bajo}')
print(f'Imágenes procesadas (Paso Alto): {processed_files_paso_alto}')
