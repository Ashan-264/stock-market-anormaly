import plotly.graph_objects as go


def create_gauge_chart(probability):
    """
    Creates a gauge chart to represent the anomaly probability.

    Args:
        probability (float): The anomaly probability (0 to 1).

    Returns:
        plotly.graph_objects.Figure: The gauge chart figure.
    """
    # Determine color based on probability
    if probability < 0.3:
        color = "green"
        label = "Low Risk"
    elif probability < 0.6:
        color = "yellow"
        label = "Moderate Risk"
    else:
        color = "red"
        label = "High Risk"

    # Create a gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"Market Anomaly Risk: {label}",
                "font": {"size": 20, "color": "black"},
            },
            number={"font": {"size": 40, "color": "black"}, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "black"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "black",
                "steps": [
                    {"range": [0, 30], "color": "rgba(0, 255, 0, 0.3)"},
                    {"range": [30, 60], "color": "rgba(255, 255, 0, 0.3)"},
                    {"range": [60, 100], "color": "rgba(255, 0, 0, 0.3)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": probability * 100,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black", "family": "Arial"},
        width=450,
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_model_probability_chart(probabilities):
    """
    Creates a bar chart showing probabilities for different models.

    Args:
        probabilities (dict): A dictionary with model names as keys and probabilities as values.

    Returns:
        plotly.graph_objects.Figure: The bar chart figure.
    """
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=probs,
                orientation="h",
                text=[f"{p:.2%}" for p in probs],
                textposition="auto",
                marker_color=["#4CAF50" if p < 0.3 else "#FFC107" if p < 0.6 else "#F44336" for p in probs],
            )
        ]
    )

    fig.update_layout(
        title={"text": "Model Probabilities", "font": {"size": 22, "color": "black"}},
        yaxis_title="Models",
        xaxis_title="Probability",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        yaxis=dict(showgrid=False, ticks="outside"),
        height=400,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "black", "family": "Arial"},
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig
