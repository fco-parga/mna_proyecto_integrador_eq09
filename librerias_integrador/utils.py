import os
import sys
import pandas as pd

def verificar_ambiente():
    # Verificar si está en Google Colab
    if 'google.colab' in sys.modules:
        print("Estas trabajando en Google Colab.")
        return 'colab'
    # Verificar si está en un sistema Windows
    elif os.name == 'nt':
        print("Estás trabajando en un sistema Windows.")
        return 'win'
    else:
        print("El ambiente de trabajo no es Google Colab ni Windows.")
        return 'other'

def cargar_y_preparar_dataframe(path_csv):
    """
    Carga un archivo CSV y realiza las siguientes transformaciones:
    - Convierte la columna 'event_timestamp' a objetos datetime.
    - Convierte la zona horaria de 'event_timestamp' a 'America/Mexico_City'.
    - Establece 'event_timestamp' como el índice del DataFrame.

    Parámetros:
    - path_csv (str): La ruta al archivo CSV que contiene los datos crudos.

    Retorna:
    - pd.DataFrame: Un DataFrame de pandas con las transformaciones aplicadas.
    """
    
    # Verificar si el path del archivo existe
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"El archivo especificado no fue encontrado: {path_csv}")
    
    # Cargar el archivo CSV
    df = pd.read_csv(path_csv, index_col=0, low_memory=False)
    
    # Convertir a datetime y ajustar zona horaria
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    df['event_timestamp'] = df['event_timestamp'].dt.tz_convert('America/Mexico_City')
    
    # Asignar como índice
    df.set_index('event_timestamp', inplace=True)
    
    return df