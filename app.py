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
    
    # The Math Formulas
    # The Math Formulas
    html.Div([
        dcc.Markdown(
            r'''
            ### The Black-Scholes Equation

            $$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$
            
            $V$: the price of the option as a function of stock price $S$ and time $t$

            $\sigma$: the volatility of the stock

            $r$: risk-free interest rate


            ### Solutions
            
            **Call Option Price:**
            $$C(S,t) = S N(d_1) - K e^{-r(T-t)} N(d_2)$$
            
            **Put Option Price:**
            $$P(S,t) = K e^{-r(T-t)} N(-d_2) - S N(-d_1)$$
            
            $$d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})(T-t)}{\sigma\sqrt{T-t}}$$
            
            $$d_2 = d_1 - \sigma\sqrt{T-t}$$
            ''',
            mathjax=True
        )
    ], style={
        'textAlign': 'center', 
        'padding': '20px', 
        'backgroundColor': '#ffffff', 
        'borderRadius': '10px', 
        'marginBottom': '20px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),

    # The Graph
    dcc.Graph(id='option-graph'),
    
    # The Controls
    html.Div([
        html.Label("Strike Price (K)"),
        dcc.Slider(
            id='slider-K', min=50, max=150, step=1, value=100, 
            marks={i: str(i) for i in range(50, 151, 10)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'  # Forces real-time updates while dragging
        ),
        
        html.Label("Risk-Free Rate (r)"),
        dcc.Slider(
            id='slider-r', min=0, max=0.20, step=0.01, value=0.05, 
            marks={0: '0%', 0.05: '5%', 0.10: '10%', 0.15: '15%', 0.20: '20%'},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
        
        html.Label("Time to Maturity (T in years)"),
        dcc.Slider(
            id='slider-T', min=0.01, max=5.0, step=0.01, value=1.0, 
            marks={i: f"{i} yr" for i in range(0, 6)},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
        
        html.Label("Volatility (σ)"),
        dcc.Slider(
            id='slider-sigma', min=0.01, max=1.0, step=0.01, value=0.20, 
            marks={0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1.0'},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ),
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'})
])

app.clientside_callback(
    """
    function(K, r, T, sigma) {
        // Create the X-axis array (Stock Prices 1 to 200)
        let S = [];
        for (let i = 1; i <= 200; i++) {
            S.push(i);
        }

        // Helper function for the Normal CDF (Approximation)
        function normCDF(x) {
            let t = 1 / (1 + 0.2316419 * Math.abs(x));
            let d = 0.3989423 * Math.exp(-x * x / 2);
            let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
            return x > 0 ? 1 - p : p;
        }

        // Arrays to hold the Y-axis results
        let calls = [];
        let puts = [];

        // Calculate Black-Scholes for every point
        let time = Math.max(T, 0.0000000001); // Prevent division by zero
        for (let i = 0; i < S.length; i++) {
            let d1 = (Math.log(S[i] / K) + (r + 0.5 * sigma * sigma) * time) / (sigma * Math.sqrt(time));
            let d2 = d1 - sigma * Math.sqrt(time);
            
            let call = S[i] * normCDF(d1) - K * Math.exp(-r * time) * normCDF(d2);
            let put = K * Math.exp(-r * time) * normCDF(-d2) - S[i] * normCDF(-d1);
            
            calls.push(call);
            puts.push(put);
        }

        // Return the Plotly figure dictionary
        return {
            'data': [
                {'x': S, 'y': calls, 'type': 'scatter', 'mode': 'lines', 'name': 'Call Price', 'line': {'color': 'green', 'width': 3}},
                {'x': S, 'y': puts, 'type': 'scatter', 'mode': 'lines', 'name': 'Put Price', 'line': {'color': 'red', 'width': 3}}
            ],
            'layout': {
                'title': 'Theoretical Option Price vs. Underlying Stock Price',
                'xaxis': {'title': 'Stock Price (S)'},
                'yaxis': {'title': 'Option Price', 'range': [-10, 160]},
                'hovermode': 'x unified',
                'template': 'plotly_white'
            }
        };
    }
    """,
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
        yaxis_range=[-10,160],
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

# Run the local server
if __name__ == '__main__':
    app.run_server(debug=True)
