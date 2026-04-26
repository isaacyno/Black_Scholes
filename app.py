import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Required for deployment

# Black-Scholes Math Function
def black_scholes(S, K, r, T, sigma):
    T = np.maximum(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put

# Application Layout
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '800px', 'margin': '0 auto'}, children=[
    html.H1("Black-Scholes Option Pricing Simulator", style={'textAlign': 'center'}),
    
    # The Graph
    dcc.Graph(id='option-graph'),
    
    # The Controls
    html.Div([
        html.Label("Strike Price (K)"),
        dcc.Slider(id='slider-K', min=50, max=150, step=1, value=100, tooltip={"placement": "bottom", "always_visible": True}),
        
        html.Label("Risk-Free Rate (r)"),
        dcc.Slider(id='slider-r', min=0, max=0.20, step=0.01, value=0.05, tooltip={"placement": "bottom", "always_visible": True}),
        
        html.Label("Time to Maturity (T in years)"),
        dcc.Slider(id='slider-T', min=0.01, max=5.0, step=0.05, value=1.0, tooltip={"placement": "bottom", "always_visible": True}),
        
        html.Label("Volatility (σ)"),
        dcc.Slider(id='slider-sigma', min=0.01, max=1.0, step=0.01, value=0.20, tooltip={"placement": "bottom", "always_visible": True}),
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'})
])

# The Callback (Updates the graph when sliders move)
@app.callback(
    Output('option-graph', 'figure'),
    [Input('slider-K', 'value'),
     Input('slider-r', 'value'),
     Input('slider-T', 'value'),
     Input('slider-sigma', 'value')]
)
def update_graph(K, r, T, sigma):
    # X-axis: Stock prices from 1 to 200
    S = np.linspace(1, 200, 200)
    
    # Calculate prices
    calls, puts = black_scholes(S, K, r, T, sigma)
    
    # Create the Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S, y=calls, mode='lines', name='Call Price', line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=S, y=puts, mode='lines', name='Put Price', line=dict(color='red', width=3)))
    
    fig.update_layout(
        title="Theoretical Option Price vs. Underlying Stock Price",
        xaxis_title="Stock Price (S)",
        yaxis_title="Option Price",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

# Run the local server
if __name__ == '__main__':
    app.run_server(debug=True)