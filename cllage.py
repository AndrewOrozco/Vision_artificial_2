import cv2
import os
import matplotlib.pyplot as plt

# Ruta de las carpetas
originals_path = 'images/Imagenes_originales'
save_path_paso_bajo = 'images/filtros-espaciales/paso-bajo'
save_path_paso_alto = 'images/filtros-espaciales/paso-alto'

# Obtener una lista de archivos en las carpetas de resultados
original_files = [f for f in os.listdir(originals_path) if f.endswith('.jpg') or f.endswith('.png')]
processed_files_paso_bajo = [f for f in os.listdir(save_path_paso_bajo) if f.endswith('.jpg') or f.endswith('.png')]
processed_files_paso_alto = [f for f in os.listdir(save_path_paso_alto) if f.endswith('.jpg') or f.endswith('.png')]

# Seleccionar dos archivos al azar para mostrar
selected_files = original_files[:2]

# Mostrar imágenes originales y procesadas
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, filename in enumerate(selected_files):
    # Cargar imágenes
    img_original = cv2.imread(os.path.join(originals_path, filename), cv2.IMREAD_GRAYSCALE)
    img_paso_bajo = cv2.imread(os.path.join(save_path_paso_bajo, 'pasobajo_' + filename), cv2.IMREAD_GRAYSCALE)
    img_paso_alto = cv2.imread(os.path.join(save_path_paso_alto, 'pasoalto_' + filename), cv2.IMREAD_GRAYSCALE)

    # Mostrar imágenes
    axs[i, 0].imshow(img_original, cmap='gray')
    axs[i, 0].set_title('Original')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(img_paso_alto, cmap='gray')
    axs[i, 1].set_title('Paso Alto')
    axs[i, 1].axis('off')


plt.tight_layout()
plt.show()

# Ruta de la carpeta de resultados del filtro Prewitt
save_path_prewitt = 'images/deteccion-bordes/Prewitt'

# Mostrar imágenes originales y procesadas
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, filename in enumerate(selected_files):
    # Cargar imágenes
    img_original = cv2.imread(os.path.join(originals_path, filename), cv2.IMREAD_GRAYSCALE)
    img_prewitt = cv2.imread(os.path.join(save_path_prewitt, 'prewitt_' + filename), cv2.IMREAD_GRAYSCALE)

    # Mostrar imágenes
    axs[i, 0].imshow(img_original, cmap='gray')
    axs[i, 0].set_title('Original')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(img_prewitt, cmap='gray')
    axs[i, 1].set_title('Prewitt')
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()

# Ruta de la carpeta de resultados del filtro Sobel
save_path_sobel = 'images/deteccion-bordes/Sobel'

# Mostrar imágenes originales y procesadas
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, filename in enumerate(selected_files):
    # Cargar imágenes
    img_original = cv2.imread(os.path.join(originals_path, filename), cv2.IMREAD_GRAYSCALE)
    img_sobel = cv2.imread(os.path.join(save_path_sobel, 'sobel_' + filename), cv2.IMREAD_GRAYSCALE)

    # Mostrar imágenes
    axs[i, 0].imshow(img_original, cmap='gray')
    axs[i, 0].set_title('Original')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(img_sobel, cmap='gray')
    axs[i, 1].set_title('Sobel')
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()

# Ruta de la carpeta de resultados del filtro Canny
save_path_canny = 'images/deteccion-bordes/Canny'

# Mostrar imágenes originales y procesadas
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, filename in enumerate(selected_files):
    # Cargar imágenes
    img_original = cv2.imread(os.path.join(originals_path, filename), cv2.IMREAD_GRAYSCALE)
    img_canny = cv2.imread(os.path.join(save_path_canny, 'canny_' + filename), cv2.IMREAD_GRAYSCALE)

    # Mostrar imágenes
    axs[i, 0].imshow(img_original, cmap='gray')
    axs[i, 0].set_title('Original')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(img_canny, cmap='gray')
    axs[i, 1].set_title('Canny')
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()

