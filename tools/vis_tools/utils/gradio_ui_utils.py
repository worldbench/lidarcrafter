import numpy as np
import plotly.graph_objs as go
from functions import sample

def generate_point_cloud():
    num_points = 100
    points = np.random.rand(num_points, 3)
    return points

def load_point_cloud_uncon(dataset, model):
    point_cloud_data = sample(dataset, model)
    return update_plot(point_cloud_data)

def load_point_cloud():
    point_cloud_data = generate_point_cloud()
    return update_plot(point_cloud_data)

def update_plot(points):
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=points[:, 2],
            colorscale='Viridis',
            opacity=1
        )
    )

    layout = go.Layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(
                showbackground=False, 
                showline=False, 
                showgrid=False,
                zeroline=False, 
                showticklabels=False,
                ticks='', 
                title=''
            ),
            yaxis=dict(
                showbackground=False,
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                ticks='',
                title=''
            ),
            zaxis=dict(
                showbackground=False,
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                ticks='',
                title='',
                range=[-4, 2]
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig = go.Figure(data=[scatter], layout=layout)
    return fig