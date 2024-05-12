import cv2 as cv
import numpy as np

def get_annotated_img_objects(img, detector_output, score_threshold=0.50, onlyhumans=True):
    """
    Anota objetos detectados en una imagen basándose en la salida de un detector.

    Parámetros:
    img (np.array): La imagen original en la que se detectarán los objetos.
    detector_output (dict): Un diccionario con las salidas del detector, incluyendo clases, cajas y puntuaciones.
    score_threshold (float, opcional): El umbral de puntuación para considerar una detección válida.
    onlyhumans (bool, opcional): Si es True, solo se anotarán detecciones humanas.

    Devuelve:
    tuple: Una tupla conteniendo la imagen anotada, el número de objetos detectados,
           los índices de las detecciones, las cajas detectadas y los tipos de objetos.
    """

    # Extrae la información relevante del output del detector
    class_ids = detector_output.get("detection_classes")[0]
    boxes = detector_output.get("detection_boxes")[0]
    scores = detector_output.get("detection_scores")[0]
    num_detections = int(detector_output.get("num_detections")[0])

    # Convierte la imagen de entrada a un array de numpy y prepara las variables
    image = np.array(img)
    objects = 0
    y_scale, x_scale, _ = image.shape
    ids = []
    object_type = []
    detected_boxes = {}

    # Decide qué objetos se van a anotar
    if onlyhumans:
        objects2plot = [1]  # Solo humanos
    else:
        objects2plot = list(set(class_ids.astype(int)))  # Todos los objetos detectados

    # Itera sobre todas las detecciones
    for i in range(num_detections):
        # Verifica si la detección cumple con el umbral y si es un objeto a anotar
        if scores[i] >= score_threshold and int(class_ids[i]) in objects2plot:
            box = boxes[i]
            y_min, x_min, y_max, x_max = box

            # Dibuja un rectángulo alrededor del objeto detectado
            cv.rectangle(image, (int(x_min*x_scale), int(y_min*y_scale)), 
                         (int(x_max*x_scale), int(y_max*y_scale)), (0, 255, 0), 5)
            objects += 1
            ids.append(i)
            object_type.append(int(class_ids[i]))
            detected_boxes[i] = [(int(x_min*x_scale), int(y_min*y_scale)), 
                                 (int(x_max*x_scale), int(y_max*y_scale))]

    # Devuelve la imagen anotada junto con información sobre las detecciones
    return image, objects, ids, detected_boxes, object_type