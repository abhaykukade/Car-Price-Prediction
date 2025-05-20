from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            cleaned = X[col].fillna('').apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])
            mlb = MultiLabelBinarizer()
            mlb.fit(cleaned)
            self.encoders[col] = mlb
        return self
    
    def transform(self, X):
        outputs = []
        for col in self.columns:
            cleaned = X[col].fillna('').apply(lambda x: [i.strip() for i in str(x).split(',') if i.strip()])
            mlb = self.encoders[col]
            # cleaned_filtered = cleaned.apply(lambda labels: [l for l in labels if l in mlb.classes_])
            transformed = mlb.transform(cleaned)
            outputs.append(transformed)
        return np.hstack(outputs)
    
    def get_feature_names_out(self):
        return [f'{col}_{cls}' for col, mlb in self.encoders.items() for cls in mlb.classes_]