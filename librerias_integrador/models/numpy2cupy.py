try:
    import cupy as cp
    cupy_library = True
except:
    cupy_library = False
    

class NumpyToCupyTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to convert NumPy arrays to CuPy arrays if CuPy is available.
    Otherwise, it returns the original NumPy array.

    Args:
        X (numpy.ndarray): Input NumPy array.

    Returns:
        numpy.ndarray or cupy.ndarray: Transformed array.
    """
    def fit(self, X, y=None):
        """
        Fit method (no learning required).

        Args:
            X (numpy.ndarray): Input data (ignored).

        Returns:
            self: The fitted transformer instance.
        """
        return self
    
    def transform(self, X):
        """
        Transform method to convert NumPy array to CuPy array (if available).

        Args:
            X (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray or cupy.ndarray: Transformed array.
        """
        if cupy_library:
            # Convert the NumPy array to a CuPy array
            return cp.asarray(X)
        else:
            # Return the same NumPy array
            return X