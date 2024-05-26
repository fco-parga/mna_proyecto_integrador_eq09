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