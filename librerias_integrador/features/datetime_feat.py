import pandas as pd

def create_date_time_features(df, timestamp_col):
    """
    Esta función toma un DataFrame y el nombre de una columna de timestamp,
    y enriquece el DataFrame con nuevas columnas que representan características
    temporales derivadas del timestamp.

    Parámetros:
    - df (pandas.DataFrame): El DataFrame que contiene la columna de timestamp.
    - timestamp_col (str): El nombre de la columna de timestamp en el DataFrame.

    Retorna:
    - df (pandas.DataFrame): El DataFrame original con las nuevas columnas de características temporales añadidas.

    Las nuevas columnas añadidas son:
    - 'year': El año extraído del timestamp.
    - 'month': El mes extraído del timestamp.
    - 'day': El día extraído del timestamp.
    - 'weekday': El día de la semana extraído del timestamp (Lunes=0, Domingo=6).
    - 'is_weekend': Un indicador binario (0 o 1) que es 1 si el día es fin de semana (sábado o domingo) y 0 en caso contrario.
    """
    # Convertir la columna de timestamp a datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extraer características de fecha y hora
    df['year'] = df[timestamp_col].dt.year
    df['month'] = df[timestamp_col].dt.month
    df['day'] = df[timestamp_col].dt.day
    df['weekday'] = df[timestamp_col].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    return df