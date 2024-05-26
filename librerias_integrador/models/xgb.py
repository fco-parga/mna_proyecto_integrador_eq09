from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from xgboost import XGBRegressor


def xgboost_pipeline(X, random_state=None, gpu=True, predict_mode=False):
    """
    Create an XGBoost pipeline for preprocessing data and applying an XGBoost model.

    Args:
        X (pandas.DataFrame): Feature matrix.
        random_state (int, optional): Seed for random number generator.
        gpu (bool, optional): Use GPU acceleration (default is True).
        predict_mode (bool, optional): Whether the pipeline is used for prediction (default is False).

    Returns:
        sklearn.pipeline.Pipeline: XGBoost pipeline.
    """
    total_cols = X.shape[1]
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numpy2cupy = NumpyToCupyTransformer()

    # Define preprocessing steps
    encoding_step = [
        ('encode_labels', OrdinalEncoder(dtype=int,
                                         handle_unknown='use_encoded_value',
                                         unknown_value=-1,
                                         encoded_missing_value=-2), categorical_columns),
    ]

    scaling_step = [
        ('scaler', RobustScaler(), list(range(total_cols)))  # Apply RobustScaler to all columns
    ]
    encoding = ColumnTransformer(encoding_step, remainder='passthrough')
    scaling = ColumnTransformer(scaling_step, remainder='drop')

    # Define the pipeline to utilize GPU or CPU
    steps = [
        ('encoding', encoding),
        ('scaling', scaling),
        ('numpy2cupy', numpy2cupy)
    ]

    if gpu and not predict_mode:
        # Define the pipeline to utilize GPU
        steps.append(('xgb_model', XGBRegressor(random_state=random_state, device="cuda")))
    else:
        # Define the pipeline for CPU
        steps.append(('xgb_model', XGBRegressor(random_state=random_state)))

    xgb_pipe = Pipeline(steps)

    return xgb_pipe