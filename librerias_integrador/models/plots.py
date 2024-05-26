import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

import matplotlib.ticker as ticker
format_eng = ticker.EngFormatter()

from librerias_integrador.models.metrics import calculate_metrics, print_metrics

def plot_cv_results(cv_results, cv_splits):
    """
    Visualiza los resultados de la validación cruzada de un modelo.

    Parámetros:
    - cv_results (dict): Un diccionario con los resultados de la validación cruzada.
    - cv_splits (KFold or int): Un objeto KFold o un entero que indica el número de divisiones.

    Salida:
    - Un gráfico de caja que muestra la distribución de los puntajes de entrenamiento y validación.
    """
    if isinstance(cv_splits, KFold):
        tot_cols = cv_splits.n_splits
    elif isinstance(cv_splits, int):
        tot_cols = cv_splits
        
    train_splits = ['split'+str(i)+'_train_score' for i in range(0, tot_cols)]
    test_splits = ['split'+str(i)+'_test_score' for i in range(0, tot_cols)]

    fig, ax = plt.subplots()

    for i in range(0, len(test_splits)):
        plt.boxplot(cv_results[train_splits[i]], labels=['cv-'+str(i)+'-train'],showmeans=True, vert=True, positions=[i-0.15], widths=0.2, patch_artist=True)    
        plt.boxplot(cv_results[test_splits[i]], labels=['cv-'+str(i)+'-val'],showmeans=True, vert=True, positions=[i+0.15], widths=0.2)

    plt.xticks(rotation=90)
    
    splits_df = pd.DataFrame(cv_results)
    print(f'Train Median per CV: {list(zip(range(0, len(test_splits)), np.round(splits_df[train_splits].median().values,4)))}')
    print(f'Test Median per CV: {list(zip(range(0, len(test_splits)), np.round(splits_df[test_splits].median().values,4)))}')
    plt.show()


def plot_predictions(ax, y_true_train, y_pred_train, y_true_test, y_pred_test, count_label, scale='linear', limit=False):
    """
    Grafica las predicciones reales vs. predichas para conjuntos de entrenamiento y prueba.

    Parámetros:
    - ax (Axes): El objeto Axes de matplotlib donde se dibujará el gráfico.
    - y_true_train (array): Valores reales del conjunto de entrenamiento.
    - y_pred_train (array): Valores predichos del conjunto de entrenamiento.
    - y_true_test (array): Valores reales del conjunto de prueba.
    - y_pred_test (array): Valores predichos del conjunto de prueba.
    - count_label (str): Etiqueta para el eje de conteo.
    - scale (str): Escala del eje ('linear' o 'log').
    - limit (bool): Si es True, limita los ejes a los valores mínimos y máximos de y_true_train.

    Salida:
    - Un gráfico de dispersión que muestra las predicciones reales vs. predichas.
    """
    ax.scatter(y_true_train, y_pred_train)
    ax.scatter(y_true_test, y_pred_test)
    min_val = min(y_true_train) if limit else 1
    max_val = max(y_true_train) * 1.005
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_xlabel(f"${count_label}_{{real}}$", fontsize=14)
    ax.set_ylabel(f"${count_label}_{{predicted}}$", fontsize=14)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_aspect('equal')


def plot_residuals(ax, y_true_train, y_pred_train, y_true_test, y_pred_test, count_label):
    """
    Grafica los residuos (diferencia entre valores reales y predichos) para conjuntos de entrenamiento y prueba.

    Parámetros:
    - ax (Axes): El objeto Axes de matplotlib donde se dibujará el histograma.
    - y_true_train (array): Valores reales del conjunto de entrenamiento.
    - y_pred_train (array): Valores predichos del conjunto de entrenamiento.
    - y_true_test (array): Valores reales del conjunto de prueba.
    - y_pred_test (array): Valores predichos del conjunto de prueba.
    - count_label (str): Etiqueta para el eje de conteo.

    Salida:
    - Un histograma que muestra la distribución de los residuos.
    """
    residuals_train = y_true_train - y_pred_train
    residuals_test = y_true_test - y_pred_test
    ax.hist(residuals_train, bins=17, label='Train')
    ax.hist(residuals_test, bins=17, label='Test')
    ax.set_title(f'Residuals: Real - Predicted ({count_label})')
    ax.set_xlabel('$\Delta$', fontsize=14)
    ax.set_ylabel('counts', fontsize=14)


def plot_model_predictions(model, X_train, X_val_test, y_train, y_val_test, counts='Clientes', 
                            nn_model=False, masking_y=None, transform_fnc=None, return_metrics=False, batch_size=16):
    """
    Realiza predicciones con el modelo proporcionado y grafica los resultados.

    Parámetros:
    - model (Model): El modelo de machine learning a utilizar para las predicciones.
    - X_train (DataFrame): Características del conjunto de entrenamiento.
    - X_val_test (DataFrame): Características del conjunto de prueba.
    - y_train (array): Valores reales del conjunto de entrenamiento.
    - y_val_test (array): Valores reales del conjunto de prueba.
    - counts (str): Etiqueta para los conteos en los gráficos.
    - nn_model (bool): Indica si el modelo es una red neuronal.
    - masking_y (int): Valor para enmascarar en las predicciones.
    - transform_fnc (Transformer): Función para revertir la transformación de los datos.
    - return_metrics (bool): Si es True, retorna las métricas calculadas.
    - batch_size (int): Tamaño del lote para la predicción en modelos de redes neuronales.

    Salida:
    - Si return_metrics es True, retorna un diccionario con las métricas calculadas.
    """

    if not nn_model:
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_val_test)
        
    if nn_model and isinstance(masking_y, int):
        y_train_pred, y_test_pred = model.predict(X_train, batch_size=batch_size), model.predict(X_val_test, batch_size=batch_size)
        
        mask_train, mask_test = np.not_equal(y_train, masking_y), np.not_equal(y_val_test, masking_y)
        
        y_train, y_train_pred = y_train.flatten()[mask_train.flatten()], y_train_pred.flatten()[mask_train.flatten()]
        y_val_test, y_test_pred = y_val_test.flatten()[mask_test.flatten()], y_test_pred.flatten()[mask_test.flatten()]

        y_train, y_train_pred = y_train.reshape(-1, 1), y_train_pred.reshape(-1, 1)
        y_val_test, y_test_pred = y_val_test.reshape(-1, 1), y_test_pred.reshape(-1, 1)

    if transform_fnc:
        y_train, y_train_pred = transform_fnc.inverse_transform(y_train), transform_fnc.inverse_transform(y_train_pred)
        y_val_test, y_test_pred = transform_fnc.inverse_transform(y_val_test), transform_fnc.inverse_transform(y_test_pred)

    metrics = {
        'train': calculate_metrics(y_train, y_train_pred),
        'test': calculate_metrics(y_val_test, y_test_pred)
    }

    print_metrics(metrics)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_predictions(axes[0], y_train, y_train_pred, y_val_test, y_test_pred, counts)
    plot_predictions(axes[1], y_train, y_train_pred, y_val_test, y_test_pred, counts, scale='log')
    plot_residuals(axes[2], y_train, y_train_pred, y_val_test, y_test_pred, counts)

    fig.tight_layout()
    plt.legend()
    plt.show()

    return metrics if return_metrics else None