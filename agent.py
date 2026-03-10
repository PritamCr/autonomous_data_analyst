from tools.eda_tool import run_eda
from tools.ml_tool import train_model, auto_ml, plot_feature_importance
from tools.query_tool import ask_question
from tools.insight_tool import generate_insights
from tools.feature_tool import suggest_features
from tools.chart_tool import correlation_heatmap, histogram_chart, scatter_chart

class DataAgent:
    """
    DataAgent wraps your dataset and provides methods for:
    - EDA
    - Charting
    - LLM insights
    - AI feature suggestions
    - ML training and AutoML
    """

    def __init__(self, df):
        self.df = df
        self.models = {}  # store trained models for feature importance

    # --------------------------
    # EDA
    # --------------------------
    def eda(self):
        """Run automated exploratory data analysis"""
        return run_eda(self.df)

    # --------------------------
    # LLM Query
    # --------------------------
    def ask(self, question):
        """Ask a question about the dataset using LLM"""
        return ask_question(self.df, question)

    # --------------------------
    # AI Insights
    # --------------------------
    def insights(self):
        """Generate high-level LLM insights"""
        return generate_insights(self.df)

    # --------------------------
    # Feature Selection
    # --------------------------
    def feature_suggestions(self, target):
        """Suggest best features using LLM for training"""
        return suggest_features(self.df, target)

    # --------------------------
    # Charts
    # --------------------------
    def heatmap(self):
        return correlation_heatmap(self.df)

    def histogram(self, column):
        return histogram_chart(self.df, column)

    def scatter(self, x_col, y_col):
        return scatter_chart(self.df, x_col, y_col)

    # --------------------------
    # ML Training
    # --------------------------
    def train(self, target, df=None, return_model=False):
        """
        Train a model using the selected features.
        Returns model type, score, and optionally the trained model.
        """
        if df is None:
            df = self.df
        if return_model:
            model_type, score, model = train_model(df, target, return_model=True)
            # Save the trained model
            self.models[target] = model
            return model_type, score, model
        else:
            model_type, score = train_model(df, target, return_model=False)
            return model_type, score

    # --------------------------
    # AutoML
    # --------------------------
    def auto_ml(self, target, df=None):
        """
        Run AutoML on the selected dataset.
        Returns all model results and the best model.
        """
        if df is None:
            df = self.df
        results, best_model = auto_ml(df, target)
        # Save the best model
        self.models[target] = best_model
        return results, best_model

    # --------------------------
    # Feature Importance
    # --------------------------
    def get_feature_importance_chart(self, target, feature_df):
        """
        Returns a Plotly chart of feature importance for tree-based models.
        Requires that a model has been trained and saved in self.models.
        """
        model = self.models.get(target)
        if model is None:
            return None
        return plot_feature_importance(model, feature_df)

    # --------------------------
    # Utility
    # --------------------------
    def get_model(self, target):
        """Return trained model object for a given target"""
        return self.models.get(target)