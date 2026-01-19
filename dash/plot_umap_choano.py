import numpy as np, dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LogNorm

# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import plotly.express as px

import seaborn as sns
import pickle

import sys
sys.path.append('../')

# save the files to investigate with the dash app
folder = "../../data/Algoriphagus/"
parquet_files = ["Expt1_251217_Pre_umap.parquet", "Expt1_251217_Post_umap.parquet"]
color_files = ['coarse_grained_Pre.parquet','coarse_grained_Post.parquet']
reduced_pts = list()
for i in range(len(parquet_files)):
    reduced_pts.append(ak.from_parquet(folder + parquet_files[i]))

color_files = list()
for i in range(len(color_files)):
    color_files.append(ak.from_parquet(folder+color_files[i]))

# ---------- 1.  build the main figure exactly as before ----------
def build_overview():
    fig = make_subplots(rows=2, cols=3)
    norms = [LogNorm(10,100), LogNorm(100,800), LogNorm(100,2000)]

    for i in range(3):
        x, y  = data[i][:,0], data[i][:,1]
        lab   = embs[i].labels

        # density (= top row)
        hist, xe, ye = np.histogram2d(x,y,bins=100)
        loghist = np.log10(hist.T+1e-3)
        fig.add_trace(go.Heatmap(
            z=loghist, x=xe, y=ye, colorscale='Viridis',
            zmin=np.log10(norms[i].vmin), zmax=np.log10(norms[i].vmax),
            colorbar=dict(x=0.3+0.35*i, len=0.4, thickness=10, y=0.82,
                          yanchor='middle', title='log10(count)')),
            row=1, col=i+1)

        # scatter (= bottom row) –─ add index in customdata
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode='markers',
            marker=dict(color=lab, size=2, colorscale='Viridis',
                        showscale=False),
            customdata=np.arange(len(x)),       # <── index here
            hovertemplate="idx %{customdata}<extra></extra>"),
            row=2, col=i+1)

    fig.update_layout(width=900, height=600, margin_t=30, showlegend=False)
    return fig
# -----------------------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(dcc.Graph(id='overview', figure=build_overview()),
             style={'display':'inline-block', 'width':'900px'}),
    html.Div(dcc.Graph(id='detail', figure=go.Figure()),
             style={'display':'inline-block', 'width':'300px', 'height':'300px',
                    'paddingLeft':'15px'})      # tiny side window
])

@app.callback(Output('detail', 'figure'),
              Input('overview', 'hoverData'))
def make_3d(hover):
    if hover is None:
        return go.Figure()

    # 2. recover dataset / point index -----------------------------
    point = hover['points'][0]
    idx   = point['customdata']   # integer index in data[i]
    col   = point['curveNumber']  # tells which subplot we came from
    i     = (col -   3) // 2      # 0,1,2 because top traces=0-2

    v = embs[i].flatten_embedding_matrix[idx]   # shape (3*L,)
    xs, ys, zs = v[0::3], v[1::3], v[2::3]

    # 3-D line / scatter ------------------------------------------
    fig = go.Figure(go.Scatter3d(x=xs, y=ys, z=zs,
                                 mode='lines+markers',
                                 marker=dict(size=3)))
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=20),
                      scene=dict(aspectmode='data'),
                      title=f'point {idx} in set {i}')
    return fig
# ---------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
