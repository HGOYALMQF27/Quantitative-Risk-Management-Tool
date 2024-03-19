# app.py

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import scipy.stats as stats
from risk_management import compute_var, compute_es, compute_cdar, mean_variance_optimization, cvar_optimization

# Generate sample data for multiple assets
np.random.seed(0)
num_assets = 5
num_days = 1000
returns = np.random.randn(num_days, num_assets)

# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1('Quantitative Risk Management Tool'),
    html.Div([
        html.Label('Select confidence level:'),
        dcc.Slider(
            id='confidence-slider',
            min=0.9,
            max=0.99,
            step=0.01,
            value=0.95,
            marks={i: str(i) for i in np.arange(0.9, 1.0, 0.01)}
        ),
        html.Label('Select target return for optimization:'),
        dcc.Input(
            id='target-return-input',
            type='number',
            value=0.05,
            step=0.01
        )
    ]),
    html.Div([
        html.Div(id='risk-metrics-output'),
        dcc.Graph(id='risk-graph')
    ]),
])

# Define callback to update risk metrics and graph
@app.callback(
    Output('risk-metrics-output', 'children'),
    [Input('confidence-slider', 'value'),
     Input('target-return-input', 'value')]
)
def update_risk_metrics(confidence_level, target_return):
    var = compute_var(returns, confidence_level)
    es = compute_es(returns, confidence_level)
    cdar = compute_cdar(returns, confidence_level=confidence_level)

    weights_mvo = mean_variance_optimization(pd.DataFrame(returns), target_return)
    weights_cvar = cvar_optimization(returns, confidence_level)

    risk_metrics_output = html.Div([
        html.Div(f'Value at Risk (VaR) at {confidence_level * 100:.2f}% confidence level: {var:.4f}'),
        html.Div(f'Expected Shortfall (ES) at {confidence_level * 100:.2f}% confidence level: {es:.4f}'),
        html.Div([html.Div(f'Conditional Drawdown at Risk (CDaR) for Asset {i+1}: {cdar[i]:.4f}') for i in range(num_assets)]),
        html.Div(f'Mean-Variance Optimization Weights: {", ".join(f"{w:.4f}" for w in weights_mvo)}'),
        html.Div(f'CVaR Optimization Weights: {", ".join(f"{w:.4f}" for w in weights_cvar)}')
    ])

    return risk_metrics_output

@app.callback(
    Output('risk-graph', 'figure'),
    [Input('confidence-slider', 'value')]
)
def update_risk_graph(confidence_level):
    portfolio_returns = np.sum(returns, axis=1)
    hist, bins = np.histogram(portfolio_returns, bins=30, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    mu, sigma = stats.norm.fit(portfolio_returns)
    pdf = stats.norm.pdf(bin_centers, mu, sigma)

    figure = {
        'data': [
            go.Bar(
                x=bins,
                y=hist,
                name='Portfolio Returns Histogram',
                marker=dict(color='blue')
            ),
            go.Scatter(
                x=bin_centers,
                y=pdf,
                mode='lines',
                name='Fitted Normal Distribution',
                line=dict(color='red', width=2)
            )
        ],
        'layout': {
            'title': 'Portfolio Returns Distribution',
            'xaxis': {'title': 'Returns'},
            'yaxis': {'title': 'Density'},
            'barmode': 'overlay'
        }
    }

    return figure

# Run Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
