import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from utils.utils import PandasSelector



class AutoSpearmanClusteringSelector(SelectorMixin, BaseEstimator):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
    
    def _correlated_pairs(self, X):
        # Compute the Spearman correlation matrix.
        corr = X.corr(method='spearman')

        # Transform into a dataframe of pairs.
        corr_df = corr.unstack().reset_index()
        corr_df.columns = ['feature_1', 'feature_2', 'correlation']

        # Apply abs() to the values.
        corr_df['correlation'] = np.abs(corr_df['correlation'])

        # Select only the pairs over the threshold.
        corr_df = corr_df[corr_df['correlation'] >= self.threshold]

        # Remove the diagonal.
        corr_df = corr_df[corr_df['feature_1'] != corr_df['feature_2']]
        
        return corr_df

    def _select_feature(self, feature_1, feature_2, X):
        others_df = X.drop(columns=[feature_1, feature_2])

        # Computes the correlation values.
        feature_1_corrs = others_df.corrwith(X[feature_1], method='spearman')
        feature_2_corrs = others_df.corrwith(X[feature_2], method='spearman')

        # Get the mean abs values.
        feature_1_corrs_mean = np.abs(feature_1_corrs).mean()
        feature_2_corrs_mean = np.abs(feature_2_corrs).mean()

        # Return the feature with the lowest value.
        if feature_1_corrs_mean <= feature_2_corrs_mean:
            return feature_1
        return feature_2

    def fit(self, X, y=None):
        # Create a selection mask with all the features as selected.
        support_mask_dict = {column: True for column in X.columns}

        # Get the pairs of correlation.
        corr_df = self._correlated_pairs(X)

        # Sort the pairs by the correlation value (descending).
        corr_df = corr_df.sort_values('correlation', ascending=False)

        # Step-wise removal of features.
        while not corr_df.empty:
            # Pop a row.
            pop_row = corr_df.iloc[0]
            corr_df = corr_df.iloc[1:]

            feature_1, feature_2 = pop_row['feature_1'], pop_row['feature_2']

            # Select the feature to keep, the one with the lower mean correlation.
            selected_feature = self._select_feature(feature_1, feature_2, X)

            # removed_feature is either feature_1 or feature_2 that was not selected.
            removed_feature = feature_2 if selected_feature == feature_1 else feature_1
            print(removed_feature)
            # Remove the feature from the pairs list.
            corr_df = corr_df[corr_df['feature_1'] != removed_feature]
            corr_df = corr_df[corr_df['feature_2'] != removed_feature]
            
            # Mark the feature in the mask as removed.
            support_mask_dict[removed_feature] = False
            
        # Convert the mask.
        self.support_mask_ = np.array([support_mask_dict[feature] for feature in X.columns])

        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, 'support_mask_')
        return self.support_mask_


class AutoSpearmanVIFSelector(SelectorMixin, BaseEstimator):
    def __init__(self, threshold=5):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Create a selection mask with all the features as selected.
        support_mask_dict = {column: True for column in X.columns}

        # Copy the dataframe to preserve the original content.
        X_copy = X.copy()

        # Step-wise remove multi-colinear features.
        while True:
            # Calculate the VIFs.
            # https://stackoverflow.com/a/48819434
            vifs = pd.Series(
                np.linalg.inv(X_copy.corr(method='spearman').to_numpy()).diagonal(),
                index=X_copy.columns,
                name='vif',
            )

            # Remove the VIFs that are below the threshold and order the Series ascending.
            cv = vifs[vifs >= self.threshold].sort_values(ascending=False)

            if len(cv) == 0:
                break

            # Get the feature to remove.
            removed_feature = cv.index[0]
            
            # Remove the feature from data frame X
            X_copy.drop(columns=removed_feature, inplace=True)

            # Mark the features in the mask as removed.
            support_mask_dict[removed_feature] = False
        
        # Convert the mask.
        self.support_mask_ = np.array([support_mask_dict[feature] for feature in X.columns])

        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, 'support_mask_')
        return self.support_mask_


class AutoSpearmanSelector(SelectorMixin, BaseEstimator):
    def __init__(self, clustering_threshold=0.7, vif_threshold=5):
        self.clustering_threshold = clustering_threshold
        self.vif_threshold = vif_threshold
    
    def fit(self, X, y=None):
        # Create the selectors.
        clustering_selector = PandasSelector(AutoSpearmanClusteringSelector(threshold=self.clustering_threshold))
        vif_selector = PandasSelector(AutoSpearmanVIFSelector(threshold=self.vif_threshold))

        # Run the selectors in parallel.
        X_clustering = clustering_selector.fit_transform(X)
        vif_selector.fit(X_clustering)

        # Compute and store the final support mask.
        selected_features = set(X_clustering.columns[vif_selector._get_support_mask()])
        self.support_mask_ = np.array([x in selected_features for x in X.columns])

        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'support_mask_')
        return self.support_mask_



class ExcludeColumnsSelector(SelectorMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        self.support_mask_ = np.array([x not in self.columns for x in X.columns])

    def _get_support_mask(self):
        check_is_fitted(self, 'support_mask_')
        return self.support_mask_
