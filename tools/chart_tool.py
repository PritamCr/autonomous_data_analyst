import plotly.express as px


def histogram_chart(df, column):

    fig = px.histogram(
        df,
        x=column,
        title=f"Histogram of {column}"
    )

    return fig


def scatter_chart(df, x_col, y_col):

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=f"{x_col} vs {y_col}"
    )

    return fig

def correlation_heatmap(df):

    numeric_df = df.select_dtypes(include="number")

    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    return fig