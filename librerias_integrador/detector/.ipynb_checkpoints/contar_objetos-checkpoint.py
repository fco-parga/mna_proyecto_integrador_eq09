from tqdm import tqdm

import cv2 as cv
import tensorflow as tf

from librerias_integrador.detector.anotar_objetos import get_annotated_img_objects

def detect_persons(img_list, detector, score_threshold=0.85, plot=True, size=(13,13)):
    """
    Detecta personas en una lista de imágenes utilizando un detector proporcionado.

    Parámetros:
    img_list (list): Lista de rutas de archivos de imágenes para procesar.
    detector (modelo): Modelo de detección de objetos para identificar personas.
    score_threshold (float, opcional): Umbral de puntuación para filtrar detecciones.
    plot (bool, opcional): Si es True, muestra las imágenes con las detecciones.
    size (tuple, opcional): Tamaño de la imagen a mostrar.

    Devuelve:
    dict: Diccionario con imágenes etiquetadas y sus cajas de detección.
    dict: Diccionario con cajas de detección por imagen.
    """

    # Inicializa diccionarios para almacenar resultados
    images_labeled = {}
    boxes_per_image = {}

    # Procesa cada imagen en la lista
    for filename in tqdm(img_list):
        # Lee la imagen y la convierte al espacio de color RGB
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Obtiene la salida del detector para la imagen actual
        detector_output = detector(tf.expand_dims(img, axis=0))
        # Anota la imagen con las detecciones de personas
        img_detect, objects, ids, detected_boxes, object_type = get_annotated_img_objects(img, detector_output, score_threshold=score_threshold)
        # Almacena la imagen anotada y las detecciones en el diccionario
        images_labeled[filename.split("\\")[-1]] = {'img': img_detect, 'boxes':detected_boxes, 'object':object_type}
        boxes_per_image[filename.split("\\")[-1]] = detected_boxes

    # Si se debe mostrar, itera sobre las imágenes etiquetadas y las muestra
    if plot:
        for name, image in images_labeled.items():
            mostrar_imagen(image['img'], cmap='', title=name, size=size)
            
    # Devuelve los diccionarios con las imágenes etiquetadas y las cajas de detección
    return images_labeled, boxes_per_image