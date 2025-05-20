import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
import warnings

def warn_unpicklable_objects(obj, name='root'):
    """
    Recursively checks whether an object (Pipeline, ColumnTransformer, custom class, etc.)
    contains anything unpicklable. Issues warnings for unpicklable parts.
    """
    def is_picklable(x):
        try:
            pickle.dumps(x)
            return True
        except Exception:
            return False

    if isinstance(obj, Pipeline):
        for step_name, step in obj.steps:
            if not is_picklable(step):
                warnings.warn(f"Pipeline step '{step_name}' is not picklable.")
            warn_unpicklable_objects(step, name=step_name)

    elif isinstance(obj, ColumnTransformer):
        for trans_name, transformer, _ in obj.transformers:
            if transformer == 'drop' or transformer == 'passthrough':
                continue
            if not is_picklable(transformer):
                warnings.warn(f"ColumnTransformer transformer '{trans_name}' is not picklable.")
            warn_unpicklable_objects(transformer, name=trans_name)

    elif isinstance(obj, TransformedTargetRegressor):
        if not is_picklable(obj.func):
            warnings.warn(f"TransformedTargetRegressor 'func' in '{name}' is not picklable.")
        if not is_picklable(obj.inverse_func):
            warnings.warn(f"TransformedTargetRegressor 'inverse_func' in '{name}' is not picklable.")
        if not is_picklable(obj.regressor):
            warnings.warn(f"TransformedTargetRegressor 'regressor' in '{name}' is not picklable.")
        warn_unpicklable_objects(obj.regressor, name=f"{name} regressor")

    elif hasattr(obj, '__dict__'):
        for attr_name, attr_value in vars(obj).items():
            if not is_picklable(attr_value):
                warnings.warn(f"Object '{name}' attribute '{attr_name}' is not picklable.")
