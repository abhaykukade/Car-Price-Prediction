from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

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


class EngineSizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.engine_map = {
            '0 - 499 cc': 250,
            '500 - 999 cc': 750,
            '1,000 - 1,999 cc': 1500,
            '2,000 - 2,999 cc': 2500,
            '3,000 - 3,999 cc': 3500,
            '4,000 - 4,999 cc': 4500,
            '5,000 - 5,999 cc': 5500,
            'More than 6,000 cc': 6500
        }

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.squeeze().map(self.engine_map).fillna(0).values.reshape(-1, 1)


class YearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.year_map = {
            'Older than 1970': 1970
        }

    def fit(self, X, y=None):
        cleaned = X.squeeze().map(self.year_map)
        num_years = pd.to_numeric(cleaned, errors='coerce')
        self.mode_ = num_years.mode().iloc[0] if not num_years.mode().empty else 1970
        return self
    
    def transform(self, X):
        cleaned = X.squeeze().map(self.year_map)
        trf = pd.to_numeric(cleaned, errors='coerce').fillna(self.mode_)
        return trf.values.reshape(-1, 1)


    # kilometers_map = {}

    # for km in train_df['kilometers'].dropna().unique():
    #     cleaned = km.replace(',', '').replace('+', '')
    #     val = cleaned.split(' - ')
    #     if len(val) > 1:
    #         km_mean = np.mean([int(val[0]), int(val[1]) + 1])
    #     else:
    #         km_mean = int(cleaned)
    #     kilometers_map[km] = km_mean
class KilometersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.km_map = {'100,000 - 109,999': 105000.0,
                        '0': 0,
                        '20,000 - 29,999': 25000.0,
                        '+200,000': 200000,
                        '190,000 - 199,999': 195000.0,
                        '40,000 - 49,999': 45000.0,
                        '50,000 - 59,999': 55000.0,
                        '150,000 - 159,999': 155000.0,
                        '170,000 - 179,999': 175000.0,
                        '130,000 - 139,999': 135000.0,
                        '110,000 - 119,999': 115000.0,
                        '60,000 - 69,999': 65000.0,
                        '80,000 - 89,999': 85000.0,
                        '30,000 - 39,999': 35000.0,
                        '90,000 - 99,999': 95000.0,
                        '120,000 - 129,999': 125000.0,
                        '10,000 - 19,999': 15000.0,
                        '1 - 999': 500,
                        '1,000 - 9,999': 5500.0,
                        '70,000 - 79,999': 75000.0,
                        '140,000 - 149,999': 145000.0,
                        '160,000 - 169,999': 165000.0,
                        '180,000 - 189,999': 185000.0
                        }

    def fit(self, X, y=None):
        self.mode_ = X.squeeze().map(self.km_map).mode()[0]
        return self
    
    def transform(self, X):
        return X.squeeze().map(self.km_map).fillna(self.mode_).values.reshape(-1, 1)
    

class BatteryCapacityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.battery_cap_map = {
            'Less than 50 kWh': 30, 
            '50 - 69 kWh': 60, 
            '70 - 89 kWh': 80, 
            '90 - 99 kWh': 95, 
            'More than 100 kWh': 120
        }

    def fit(self, X, y=None):
        self.mode_ = X.squeeze().map(self.battery_cap_map).mode()[0]
        return self
    
    def transform(self, X):
        return X.squeeze().map(self.battery_cap_map).fillna(self.mode_).values.reshape(-1, 1)
    

class BatteryRangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.battery_range_map = {
            '400 - 499 km': 450, 
            '300 - 399 km': 350, 
            'More than 500 km': 550, 
            '200 - 299 km': 250, 
            '100 - 199 km': 150, 
            'Less than 100 km': 50
        }

    def fit(self, X, y=None):
        self.mode_ = X.squeeze().map(self.battery_range_map).mode()[0]
        return self
    
    def transform(self, X):
        return X.squeeze().map(self.battery_range_map).fillna(self.mode_).values.reshape(-1, 1)
    
