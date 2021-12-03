import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skrvm import RVR

class ROIsFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the 284 ROIs features:"""
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, :284]


class VBMFeatureExtractor(BaseEstimator, TransformerMixin):
    """Select only the last 284 ROIs features:"""
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[:, 284:]


def get_estimator():
    """Build your estimator here."""
    
    estimator = make_pipeline(VBMFeatureExtractor(), StandardScaler(), 
                              ElasticNet(alpha =0.1,l1_ratio= 0.1, max_iter = 30000, tol=1e-2))
    return estimator
    
    
    
    
    
    
    