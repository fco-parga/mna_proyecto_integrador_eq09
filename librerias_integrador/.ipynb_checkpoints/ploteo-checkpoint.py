import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_df_counts_histogram(raw_df, column='people', normalize=False):

    df_counts = pd.DataFrame(raw_df[column].value_counts(normalize=normalize)).reset_index().sort_values(column).reset_index(drop=True)
    if normalize:
        column2display = 'proportion'
    else:
        column2display = 'count'
        
    # Crear un gráfico de barras
    plt.figure(figsize=(5, 3))
    plt.bar(df_counts[column], df_counts[column2display], color='skyblue', edgecolor='black', align='center')
    
    # Añadir títulos y etiquetas
    plt.title('Conteo de Personas')
    plt.xlabel('Número de Personas')
    plt.ylabel('Conteo')
    plt.xticks(range(0, 11))  # Asegura que el eje x tenga marcas de 0 a 10
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Mostrar el gráfico
    plt.show()


def plot_df_counts_timeseries(df, figsize=(13, 3), grouped=False, agg2plot=None, intervalo=''):
    """
    Genera un gráfico de líneas del conteo de personas a lo largo del tiempo.

    Parámetros:
    df (DataFrame): DataFrame de pandas que contiene los datos a graficar.
    figsize (tuple, opcional): Tamaño de la figura del gráfico.
    grouped (bool, opcional): Indica si los datos están agrupados por alguna métrica.
    agg2plot (str, opcional): Nombre de la columna agregada a graficar si los datos están agrupados.
    intervalo (str, opcional): Intervalo de tiempo de los datos para incluir en el título.

    Devuelve:
    None: Esta función no devuelve nada, solo muestra el gráfico.
    """

    # Configura el tamaño de la figura del gráfico
    plt.figure(figsize=figsize)
    
    # Grafica los datos agrupados o el conteo de personas por captura
    if grouped and agg2plot:
        plt.plot(df.index, df[agg2plot], '-o', markersize=2, label='Conteo')  
        metrica = agg2plot
    else:
        plt.plot(df.index, df['people'], '-o', markersize=2, label='Conteo')   
        plt.plot(df.index, np.zeros(df.shape[0]), 'o', markersize=2, color='darkgreen', label='Capturas')  
        metrica = 'por captura'
    
    # Configura el formato de las etiquetas del eje x para fechas
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    
    # Ajusta la orientación de las etiquetas de fecha
    plt.gcf().autofmt_xdate()
    
    # Establece las etiquetas y el título del gráfico
    plt.xlabel('Fecha')
    plt.ylabel('Número de personas')
    titulo = f'Conteo de personas ({metrica}) {intervalo}'
    plt.title(titulo)
    
    # Activa la grilla para mejor visualización
    plt.grid(True)
    # Muestra la leyenda del gráfico
    plt.legend()
    
    # Muestra el gráfico
    plt.show()