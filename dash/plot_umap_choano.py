import numpy as np, dash
from dash import dcc, html, Input, Output, MATCH, ctx
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
import h5py
import pickle

import sys
sys.path.append('../')



xml_files = ["Expt1_251217_Pre", "Expt1_251217_Post"]

data, colors, embs, mkvs = [], [], [], []

input_filename = '../data/choano/interim/nested_data.h5'
#input_filename = '../data/choano/interim/nested_multi_embs.h5'

with h5py.File(input_filename, 'r') as hf:
    for name in xml_files:
        # Access the group
        grp = hf[name]
        
        # Load Arrays (Note: [:] loads the data into memory)
        data.append(grp['umap_points'][:])
        colors.append(grp['color'][:])
        
        # Load Pickles
        # We read the void data -> bytes -> unpickle
        # .tobytes() converts the numpy void wrapper back to raw bytes
        emb_blob = grp['embedding'][()] 
        embs.append(pickle.loads(emb_blob.tobytes()))
        
        mkv_blob = grp['markov'][()]
        mkvs.append(pickle.loads(mkv_blob.tobytes()))

print("Data loaded successfully.")

Dimension = embs[0].D_total


# ... (Imports and Data Loading remain the same) ...

# Mock data for demonstration if files are missing
# data = [np.random.rand(100, 2) for _ in range(2)]
# embs = [type('obj', (object,), {'flatten_embedding_matrix': np.random.rand(100, 30)})() for _ in range(2)]
# Dimension = 3

# ---------- 1. Helper to build a SINGLE overview figure ----------
def build_single_overview(dataset_index):
    # Extract data for this specific index
    xy = data[dataset_index]
    x, y = xy[:, 0], xy[:, 1]
    
    # Create a standalone figure (not a subplot)
    fig = go.Figure(go.Scattergl(
        x=x, y=y, 
        mode='markers',
        marker=dict(size=3, color=colors[dataset_index], colorscale='Viridis', showscale=False), # Mock color
        # We don't need customdata; we will use pointIndex
        hovertemplate="Index: %{pointIndex}<extra></extra>"
    ))
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,  # Fixed height for consistency
        title=f'Dataset {dataset_index}'
    )
    return fig

app = dash.Dash(__name__)

# ---------- 2. Dynamic Layout Generation ----------
def create_app_layout():
    # We create a list of columns. 
    # Each column holds one 'Overview' and one 'Detail' for index 'i'
    graph_columns = []
    
    for i in range(len(data)):
        col = html.Div([
            # TOP GRAPH (Overview)
            dcc.Graph(
                # ID is a DICTIONARY now, allowing pattern matching
                id={'type': 'overview', 'index': i}, 
                figure=build_single_overview(i),
                style={'height': '500px'}
            ),
            
            # BOTTOM GRAPH (Detail) - aligned directly under
            dcc.Graph(
                id={'type': 'detail', 'index': i},
                figure=go.Figure(), # Start empty
                style={'height': '500px'}
            )
        ], style={
            'flex': '1',        # Distribute space equally
            'minWidth': '250px', # Prevent squishing
            'padding': '5px'
        })
        graph_columns.append(col)

    # Return a Flexbox container holding these columns
    return html.Div(graph_columns, style={'display': 'flex', 'flexDirection': 'row'})

app.layout = create_app_layout()


# ---------- 3. The Pattern-Matching Callback ----------
@app.callback(
    Output({'type': 'detail', 'index': MATCH}, 'figure'),
    Input({'type': 'overview', 'index': MATCH}, 'hoverData')
)
def update_detail(hover):
    # 1. Determine which dataset triggered this callback
    # ctx.triggered_id returns the dictionary ID of the input that fired
    if not ctx.triggered_id:
        return dash.no_update
    
    i = ctx.triggered_id['index']  # This gets the 'i' automatically!

    if hover is None:
        return go.Figure()

    point = hover['points'][0]

    if 'pointIndex' not in point:
        return dash.no_update

    idx = point['pointIndex']

    # Safety Check
    if i >= len(embs):
        return dash.no_update

    # 2. Retrieve Data using 'i' and 'idx'
    try:
        v = embs[i].flatten_embedding_matrix[idx]
    except IndexError:
        return dash.no_update

    # 3. Build Figure (Logic from before)
    if Dimension == 3:
        xs, ys, zs = v[0::3], v[1::3], v[2::3]
        fig = go.Figure(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines+markers', marker=dict(size=3)))
        fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=30))
    
    elif Dimension == 2:
        xs, ys = v[0::2], v[1::2]
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode='lines+markers', marker=dict(size=2)))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))

        # 2D LIMITS
        fig.update_layout(
            xaxis=dict(range=[-100, 100], constrain='domain'),
            yaxis=dict(range=[-100, 100], scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, b=0, t=30)
        )

    fig.update_layout(title=f'ID {idx} (Set {i})')
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8051)