import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

# Example data
x = np.random.randn(1000)
y = np.random.randn(1000)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H4("Hover over a point to see zoomed view"),
    dcc.Graph(id='main-plot'),
    dcc.Graph(id='mini-plot')
])

@app.callback(
    Output('mini-plot', 'figure'),
    Input('main-plot', 'hoverData')
)
def update_mini_plot(hoverData):
    if hoverData is None:
        return go.Figure()

    point = hoverData['points'][0]
    xi, yi = point['x'], point['y']

    # Simulate a local zoomed plot or something related
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[xi - 0.2, xi, xi + 0.2],
        y=[yi - 0.2, yi, yi + 0.2],
        mode='markers+lines',
        marker=dict(size=8)
    ))
    fig.update_layout(title=f"Detail around ({xi:.2f}, {yi:.2f})")
    return fig

@app.callback(
    Output('main-plot', 'figure'),
    Input('main-plot', 'id')  # dummy input to trigger once
)
def plot_main(_):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode='markers',
        marker=dict(size=5)
    ))
    return fig

if __name__ == '__main__':
    app.run(debug=True)
