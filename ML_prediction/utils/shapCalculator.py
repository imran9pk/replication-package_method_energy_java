import shap
import matplotlib.pyplot as plt

def calculateShap(model, X_test):
     explainer = shap.Explainer(model)
     shap_values = explainer(X_test)#, plot_type='bar'
     shap.summary_plot(shap_values[:, :, 0], feature_names=X_test.columns, max_display=44, plot_type='bar')


     plt.show()