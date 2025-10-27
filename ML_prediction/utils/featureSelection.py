from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SequentialFeatureSelector, RFE, RFECV, VarianceThreshold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SequentialFeatureSelector

from utils.conf_selectors import AutoSpearmanSelector
from utils.utils import PandasTransformer, PandasSelector

from sklearn.feature_selection import SelectKBest, f_regression, chi2
import pandas as pd


os = SMOTE(random_state=42)

def sequentialSelection(X_Data, Y_Data, task_type):
    print(X_Data.columns)

    if task_type == 'classification':
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        scoring = 'f1'
    else:
        model = DecisionTreeRegressor(random_state=42)
        scoring = 'r2'  # or 'neg_mean_squared_error', depending on your preference

    sfs = SequentialFeatureSelector(model, direction="backward", n_features_to_select=1, scoring=scoring)
    sfs.fit(X_Data, Y_Data)
    print(sfs.get_feature_names_out(X_Data.columns))

def rfe(X_Data, Y_Data, task_type):
    if task_type == 'classification':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
    else:
        model = RandomForestRegressor(random_state=42)  # no class_weight in regressors

    selector = RFE(model, n_features_to_select=5, step=1)
    selector = selector.fit(X_Data, Y_Data)
    print(selector.support_)
    print(selector.ranking_)

def rfecv(X_Data, Y_Data, task_type):
    if task_type == 'classification':
        model = DecisionTreeClassifier(random_state=42)
        scoring = 'f1'
    else:
        model = DecisionTreeRegressor(random_state=42)
        scoring = 'r2'  # or 'neg_mean_squared_error'

    selector = RFECV(model, step=1, cv=5, scoring=scoring)
    selector = selector.fit(X_Data, Y_Data)
    print(selector.support_)
    print(selector.get_feature_names_out(X_Data.columns))
    return selector.get_feature_names_out(X_Data.columns)

def autoSpearman(X_Data):
    preprocess_pipeline = Pipeline([
        ('scaler', PandasTransformer(StandardScaler())),
        ('selector1', PandasSelector(VarianceThreshold())),
        ('selector2', PandasSelector(AutoSpearmanSelector(clustering_threshold=0.7, vif_threshold=10))),
    ])
    # Fit on the train.
    # print(len(X_train))
    X_Data_Reduced = preprocess_pipeline.fit_transform(X_Data)
    # print(X_train[0])
    #X_test_reduced = preprocess_pipeline.fit_transform(X_Test)

    return X_Data_Reduced

def select_features_kbest(X, y, task_type, k=10):

    if task_type == 'classification':
        selector = SelectKBest(score_func=chi2, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)
    
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_columns), selected_columns