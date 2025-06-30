from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from utils.utils import PandasTransformer, PandasSelector
from utils.conf_selectors import AutoSpearmanSelector
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV

def autoSpearman(X_Data):
    preprocess_pipeline = Pipeline([
        ('scaler', PandasTransformer(StandardScaler())),
        ('selector1', PandasSelector(VarianceThreshold())),
        ('selector2', PandasSelector(AutoSpearmanSelector(clustering_threshold=0.7, vif_threshold=10))),
    ])

    X_Data_Reduced = preprocess_pipeline.fit_transform(X_Data)

    return X_Data_Reduced

def rfecv(X_Data, Y_Data):
    regressor = DecisionTreeRegressor(random_state=42)
    
    # Use R² or neg_mean_squared_error for scoring
    selector = RFECV(regressor, step=1, cv=5, scoring="r2")
    selector = selector.fit(X_Data, Y_Data)
    
    print(selector.support_)
    print(selector.get_feature_names_out(X_Data.columns))
    
    return selector.get_feature_names_out(X_Data.columns)
