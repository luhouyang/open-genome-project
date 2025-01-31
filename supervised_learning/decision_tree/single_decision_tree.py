# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

plot_tree(clf)
# %%

# %%
import shap

explainer = shap.Explainer(clf)
shap_values = explainer(X)

shap.plots.waterfall(shap_values[0, :, 0])
# %%
