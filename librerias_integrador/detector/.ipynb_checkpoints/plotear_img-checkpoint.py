import matplotlib.pyplot as plt

def plot_images_in_grid(images_labeled, columns=3, cmap='gray', size=(23, 23)):
    """
    Muestra una cuadrícula de imágenes con anotaciones.

    Parámetros:
    images_labeled (dict): Un diccionario con nombres de imágenes y sus datos correspondientes.
    columns (int, opcional): Número de columnas en la cuadrícula.
    cmap (str, opcional): Mapa de colores para mostrar las imágenes.
    size (tuple, opcional): Tamaño de la figura en pulgadas.

    Devuelve:
    None: Esta función no devuelve nada, solo muestra las imágenes en una cuadrícula.
    """

    # Calcula el número total de imágenes
    total_images = len(images_labeled)
    
    # Calcula el número de filas necesarias
    rows = total_images // columns + (total_images % columns > 0)
    
    # Configura el tamaño de la figura basado en el número de filas y columnas
    plt.figure(figsize=(size[0], size[1] * rows))
    
    # Itera a través de las imágenes y sus nombres
    for index, (name, image) in enumerate(images_labeled.items()):
        # Crea un subplot para cada imagen
        plt.subplot(rows, columns, index + 1)
        # Muestra la imagen con o sin un mapa de colores
        if cmap == "":
            plt.imshow(image['img'])
        else:
            plt.imshow(image['img'], cmap)
        # Cuenta el número de objetos detectados
        n_counts = len(image['object'])
        # Define el título con el nombre y el número de personas detectadas
        title = f'{name}\nPersonas detectadas: {n_counts}'
        plt.title(title, fontsize=10)
        plt.axis('off')  # Oculta los ejes
    
    plt.tight_layout()  # Ajusta la disposición de los subplots
    plt.show()  # Muestra la cuadrícula con todas las imágenes
