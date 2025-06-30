import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin



class PandasTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer
    
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        X_transformed = self.transformer.transform(X, y)
        return pd.DataFrame(X_transformed, columns=X.columns)


class PandasSelector(SelectorMixin, BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def _get_support_mask(self):
        return self.transformer._get_support_mask()

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self
    
    def transform(self, X):
        X_transformed = self.transformer.transform(X)
        columns = X.columns[self.transformer.get_support()]
        return pd.DataFrame(X_transformed, columns=columns)


def remove_negative_values(df, dep_var):
    return df[df[dep_var] >= 0].reset_index(drop=True)


def approximate_zeros(df, dep_var, replacement=0.001):
    df = df.copy()
    df.loc[df[dep_var] == 0, dep_var] = replacement
    return df


def apply_binary_threshold(y, threshold):
    return pd.cut(y, bins=[0, threshold, np.inf], labels=[0, 1])
