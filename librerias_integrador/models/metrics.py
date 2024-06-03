import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error, make_scorer

def calculate_metrics(y_true, y_pred):
    """
    Calcula varias métricas de rendimiento para comparar los valores reales y predichos.

    Parámetros:
    - y_true (array): Valores reales.
    - y_pred (array): Valores predichos.

    Salida:
    - Un diccionario con las métricas calculadas: RMSE, MAE, MAPE y R^2.
    """
    return {
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_true > 0),
        'r2': r2_score(y_true, y_pred)
    }


def print_metrics(metrics):
    """
    Imprime las métricas de rendimiento de manera legible.

    Parámetros:
    - metrics (dict): Un diccionario con las métricas a imprimir.

    Salida:
    - Impresión en consola de las métricas de rendimiento.
    """
    for label, values in metrics.items():
        print(f'\n{label}:')
        for metric, value in values.items():
            print(f'  {metric}: {value:.4f}', end=', ')
        print('\b\b')  # Esto elimina la última coma y espacio


def neg_root_mean_squared_error(y_true, y_pred):
    return -1 * root_mean_squared_error(y_true, y_pred)


def plot_metrics(metricas_por_modelo_df, titles, nrows=2, ncols=2, figsize=(14, 5)):
    """
    Genera un grid de gráficos scatter plot para las métricas de modelos de machine learning.

    Parámetros:
    - metricas_por_modelo_df (DataFrame): Un DataFrame con las métricas de los modelos.
    - titles (list): Una lista de títulos de métricas a graficar.

    Salida:
    - Cuatro gráficos scatter plot en un grid de 2x2.
    """
    # Crear figuras y ejes para un grid de 2x2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Iterar sobre cada métrica y crear un gráfico para cada una
    for i, metric in enumerate(titles):
        ax = axes[i//2, i%2]
        # Filtrar los datos de entrenamiento y prueba para la métrica actual
        train_data = metricas_por_modelo_df.xs('train', level='dataset')[metric]
        test_data = metricas_por_modelo_df.xs('test', level='dataset')[metric]
        
        # Obtener los nombres de los modelos para el eje x
        x_labels = train_data.index
        
        # Crear scatter plot para los datos de entrenamiento y prueba
        ax.scatter(x_labels, train_data, marker='o', label='Train')
        ax.scatter(x_labels, test_data, marker='x', label='Test')
        
        ax.plot(x_labels, train_data, color='blue', linestyle='-', linewidth=0.5)
        ax.plot(x_labels, test_data, color='orange', linestyle='-', linewidth=0.5)
        
        # Establecer el título y las etiquetas
        ax.set_title(metric)
        ax.set_ylabel(metric)
        
        # Añadir leyenda
        ax.legend()

    # Ajustar el layout y mostrar el gráfico
    plt.tight_layout()
    plt.show()