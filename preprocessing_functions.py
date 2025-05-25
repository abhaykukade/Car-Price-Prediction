import numpy as np

def convert_to_str(X):
    return X.astype(str)

def log_transform(x):
    return np.log1p(x)

def exp_transform(x):
    return np.expm1(x)

