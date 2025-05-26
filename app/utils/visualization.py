import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def create_temporal_chart(data: pd.Series, title: str, x_label: str, y_label: str) -> go.Figure:
    fig = px.line(
        x=data.index,
        y=data.values,
        title=title,
        labels={'x': x_label, 'y': y_label}
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_horizontal_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                               title: str, color_scale: str = 'viridis') -> go.Figure:
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation='h',
        title=title,
        color=x_col,
        color_continuous_scale=color_scale
    )
    
    fig.update_layout(
        height=max(400, len(df) * 25),
        showlegend=False
    )
    
    return fig

def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                       title: str, hover_data: List[str] = None,
                       log_x: bool = False, log_y: bool = False) -> go.Figure:
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=title,
        hover_data=hover_data or [],
        log_x=log_x,
        log_y=log_y,
        opacity=0.6
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_correlation_heatmap(corr_matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(
        corr_matrix,
        title=title,
        color_continuous_scale='RdBu',
        aspect='auto',
        text_auto=True
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_distribution_plot(data: pd.Series, title: str, 
                           nbins: int = 30, log_x: bool = False) -> go.Figure:
    fig = px.histogram(
        x=data,
        title=title,
        nbins=nbins,
        log_x=log_x
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_multi_line_chart(data_dict: Dict[str, pd.Series], title: str, 
                          x_label: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    
    for name, data in data_dict.items():
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines+markers',
            name=name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                     color_col: str = None, title: str = "") -> go.Figure:
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        title=title,
        opacity=0.7
    )
    
    fig.update_layout(height=700)
    
    return fig
