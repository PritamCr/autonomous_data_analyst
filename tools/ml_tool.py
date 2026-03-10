from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd

def train_model(df, target, return_model=False):
    """
    Train a model on the selected features.

    Parameters:
    - df: pandas DataFrame (selected features + target)
    - target: string, name of the target column
    - return_model: bool, whether to return the trained model object

    Returns:
    - model_type: "Classification" or "Regression"
    - score: accuracy (classification) or R2 (regression)
    - model (optional): trained model object
    """

    # Drop target to get features
    X = df.drop(columns=[target])
    y = df[target]

    # Keep only numeric columns
    X = X.select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric features selected for training.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Detect problem type
    if y.nunique() < 10:  # Classification
        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42)
        }
        best_score = 0
        best_model = None
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            if score > best_score:
                best_score = score
                best_model = model
        if return_model:
            return "Classification", best_score, best_model
        return "Classification", best_score

    else:  # Regression
        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "LinearRegression": LinearRegression()
        }
        best_score = float("-inf")
        best_model = None
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            if score > best_score:
                best_score = score
                best_model = model
        if return_model:
            return "Regression", best_score, best_model
        return "Regression", best_score


def auto_ml(df, target):
    """
    AutoML: Train multiple models and return all results + best model.

    Returns:
    - results: list of tuples [(model_name, score), ...]
    - best_model: tuple (model_name, score)
    """
    X = df.drop(columns=[target]).select_dtypes(include="number")
    y = df[target]

    if X.shape[1] == 0:
        raise ValueError("No numeric features for AutoML.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    # Classification
    if y.nunique() < 10:
        models = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42)
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            results.append((name, score))
    else:  # Regression
        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "LinearRegression": LinearRegression()
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            results.append((name, score))

    best_model = max(results, key=lambda x: x[1])
    return results, best_model


def plot_feature_importance(model, X):
    """
    Returns a Plotly bar chart of feature importances.
    Works only for tree-based models.
    """
    import plotly.express as px
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_importance = pd.DataFrame({"feature": X.columns, "importance": importances})
        df_importance = df_importance.sort_values("importance", ascending=False)
        fig = px.bar(df_importance, x="feature", y="importance", title="Feature Importance")
        return fig
    else:
        return None