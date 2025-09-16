import numpy as np, dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LogNorm

# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import plotly.express as px

import seaborn as sns
import pickle

import sys
sys.path.append('../')

name = 'straight_helix'



data = np.load('../data/toy_model/processed/umap_points_'+name+'.npy')
indice = np.load('../data/toy_model/processed/umap_indices_'+name+'.npy')
with open('../data/toy_model/processed/embedding_'+name+'.pkl', "rb") as f:
    emb = pickle.load(f)
with open('../data/toy_model/processed/markov_'+name+'.pkl', "rb") as f:
    mkv = pickle.load(f)
cluster_center = np.load('../data/toy_model/processed/umap_centers_'+name+'.npy')

# ---- 1) Overview figure: don't hard-code width/height here ----
def build_overview():
    fig = make_subplots()
    x, y = data[:,0], data[:,1]
    lab  = emb.labels
    fig.add_trace(go.Scattergl(
        x=x, y=y, mode='markers',
        marker=dict(color=lab, size=2, colorscale='Viridis', showscale=False),
        customdata=np.arange(len(x)),
        hovertemplate="idx %{customdata}<extra></extra>"
    ))
    fig.update_layout(margin_t=30, showlegend=False, uirevision="keep")
    return fig

app = dash.Dash(__name__)

# ---- 2) Layout: side-by-side, same size, top-aligned ----
app.layout = html.Div([
    dcc.Graph(id='overview', figure=build_overview(),
              style={'flex':'1 1 0%', 'height':'600px'}),
    dcc.Graph(id='detail',
              style={'flex':'1 1 0%', 'height':'600px'})
], style={'display':'flex', 'alignItems':'flex-start', 'gap':'16px'})


# ---- 3) Callback: equal padded ranges for the 3-D plot ----
@app.callback(Output('detail', 'figure'), Input('overview', 'hoverData'))
def make_3d(hover):
    if hover is None:
        return go.Figure(layout=dict(uirevision="keep"))

    idx = hover['points'][0]['customdata']
    v = emb.flatten_embedding_matrix[idx]   # shape (3*L,)
    xs, ys, zs = v[0::3], v[1::3], v[2::3]

    # center + equal span with padding (handles straight lines)
    mins = np.array([xs.min(), ys.min(), zs.min()])
    maxs = np.array([xs.max(), ys.max(), zs.max()])
    ctr  = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    if span == 0: span = 1.0
    pad = 0.05 * span
    half = (span/2.0) + pad

    xr = [ctr[0]-half, ctr[0]+half]
    yr = [ctr[1]-half, ctr[1]+half]
    zr = [ctr[2]-half, ctr[2]+half]

    fig = go.Figure(go.Scatter3d(x=xs, y=ys, z=zs,
                                 mode='lines+markers',
                                 marker=dict(size=3)))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        title=f'point {idx}',
        scene=dict(
            aspectmode='cube',               # equal axis scales
            xaxis=dict(range=xr, zeroline=False),
            yaxis=dict(range=yr, zeroline=False),
            zaxis=dict(range=zr, zeroline=False),
        ),
        uirevision="keep"                   # keep camera between updates
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)