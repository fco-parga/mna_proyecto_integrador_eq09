import numpy as np

class DummyModel:
    def __init__(self):
        """Inicializa el modelo con la media establecida en None."""
        self.mean = None

    def fit(self, X, y):
        """
        Ajusta el modelo a los datos de entrenamiento.
        
        Calcula la media del arreglo 'y' y la almacena en la variable de instancia.
        El DataFrame 'X' se proporciona para cumplir con la interfaz común, pero no se utiliza.
        
        Parámetros:
        - X: DataFrame de pandas (no utilizado en el cálculo).
        - y: Arreglo de NumPy 1D con los valores objetivo.
        """
        self.mean = np.mean(y)

    def predict(self, X_test):
        """
        Predice utilizando el modelo.
        
        Devuelve un arreglo lleno con la media de los datos de entrenamiento, con la misma longitud
        que el arreglo 'X_test' proporcionado.
        
        Parámetros:
        - X_test: DataFrame de pandas con los datos sobre los cuales se desea hacer la predicción.
        
        Devuelve:
        - Un arreglo de NumPy lleno con la media de los datos de entrenamiento.
        """
        return np.full(len(X_test), self.mean)