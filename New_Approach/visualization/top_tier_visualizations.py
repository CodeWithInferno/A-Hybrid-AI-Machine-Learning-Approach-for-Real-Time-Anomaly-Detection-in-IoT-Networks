"""
Advanced Visualization Library for Temporal Deception Detection in IoT Networks
==============================================================================

A comprehensive visualization toolkit for research-grade temporal analysis of IoT attacks,
featuring interactive 3D plots, multi-scale temporal analysis, and publication-ready styling.

Author: IoT Security Research Lab
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.preprocessing import StandardScaler
import colorcet as cc

warnings.filterwarnings('ignore')

# Custom color schemes for publication quality
CUSTOM_COLORS = {
    'normal': '#2E86AB',
    'attack': '#E63946',
    'transition': '#F77F00',
    'background': '#F1F3F4',
    'grid': '#E0E0E0',
    'text': '#2D3436',
    'accent': '#6C5CE7'
}

# Research-grade color palettes
TEMPORAL_PALETTE = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
ATTACK_PALETTE = ['#03045E', '#023E8A', '#0077B6', '#0096C7', '#00B4D8', '#48CAE4', '#90E0EF', '#ADE8F4', '#CAF0F8']
HEATMAP_COLORSCALE = [[0, '#1a1a2e'], [0.5, '#16213e'], [0.75, '#e94560'], [1, '#f47068']]


class TemporalDeceptionVisualizer:
    """
    Advanced visualization class for temporal deception detection research.
    Provides publication-quality plots with interactive features.
    """
    
    def __init__(self, style: str = 'research'):
        """
        Initialize the visualizer with custom styling.
        
        Args:
            style (str): Visual style preset ('research', 'dark', 'minimal')
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """Configure matplotlib and plotly styling for publication quality."""
        if self.style == 'research':
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette(TEMPORAL_PALETTE)
        elif self.style == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('seaborn-v0_8-white')
            
        # Set high DPI for publication quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        
    def create_3d_temporal_heatmap(self, 
                                   data: pd.DataFrame,
                                   time_col: str,
                                   feature_cols: List[str],
                                   value_col: str,
                                   title: str = "3D Temporal Attack Heatmap") -> go.Figure:
        """
        Create an advanced 3D temporal heatmap with plotly.
        
        Args:
            data: DataFrame containing temporal data
            time_col: Column name for time dimension
            feature_cols: List of feature column names
            value_col: Column name for values to visualize
            title: Plot title
            
        Returns:
            Plotly figure object
            
        Example:
            >>> viz = TemporalDeceptionVisualizer()
            >>> fig = viz.create_3d_temporal_heatmap(df, 'timestamp', ['cpu', 'memory'], 'anomaly_score')
            >>> fig.show()
        """
        # Prepare data for 3D visualization
        time_bins = pd.cut(data[time_col], bins=50)
        grouped = data.groupby([time_bins] + feature_cols)[value_col].mean().reset_index()
        
        # Create meshgrid for 3D surface
        x = np.unique(grouped[feature_cols[0]])
        y = np.unique(grouped[time_col])
        z = grouped[value_col].values.reshape(len(y), len(x))
        
        fig = go.Figure(data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale=HEATMAP_COLORSCALE,
                contours={
                    "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project": {"z": True}}
                },
                hovertemplate='%{x}<br>Time: %{y}<br>Value: %{z:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 24, 'family': 'Arial Black'}
            },
            scene=dict(
                xaxis_title=feature_cols[0],
                yaxis_title="Time",
                zaxis_title=value_col,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                ),
                xaxis=dict(gridcolor='rgb(255, 255, 255)', gridwidth=2),
                yaxis=dict(gridcolor='rgb(255, 255, 255)', gridwidth=2),
                zaxis=dict(gridcolor='rgb(255, 255, 255)', gridwidth=2),
            ),
            width=1000,
            height=800,
            paper_bgcolor=CUSTOM_COLORS['background'],
            font=dict(family="Arial", size=12, color=CUSTOM_COLORS['text'])
        )
        
        return fig
    
    def animated_attack_evolution(self,
                                 data: pd.DataFrame,
                                 time_col: str,
                                 x_col: str,
                                 y_col: str,
                                 category_col: str,
                                 title: str = "Attack Evolution Over Time") -> go.Figure:
        """
        Create animated scatter plot showing attack evolution over time.
        
        Args:
            data: DataFrame with temporal attack data
            time_col: Column for animation frames
            x_col, y_col: Columns for x and y axes
            category_col: Column for color coding (e.g., attack type)
            title: Plot title
            
        Returns:
            Animated plotly figure
        """
        # Sort data by time
        data_sorted = data.sort_values(time_col)
        
        fig = px.scatter(
            data_sorted,
            x=x_col,
            y=y_col,
            animation_frame=time_col,
            animation_group=category_col,
            color=category_col,
            size='anomaly_score' if 'anomaly_score' in data.columns else None,
            hover_name=category_col,
            color_discrete_map={'normal': CUSTOM_COLORS['normal'], 'attack': CUSTOM_COLORS['attack']},
            range_x=[data[x_col].min()*0.9, data[x_col].max()*1.1],
            range_y=[data[y_col].min()*0.9, data[y_col].max()*1.1]
        )
        
        fig.update_traces(
            marker=dict(line=dict(width=1, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )
        
        fig.update_layout(
            title={'text': title, 'font': {'size': 26, 'family': 'Arial Black'}},
            xaxis=dict(title=x_col, gridcolor=CUSTOM_COLORS['grid']),
            yaxis=dict(title=y_col, gridcolor=CUSTOM_COLORS['grid']),
            paper_bgcolor=CUSTOM_COLORS['background'],
            plot_bgcolor='white',
            font=dict(family="Arial", size=14),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            width=1000,
            height=600
        )
        
        # Add play button styling
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
        
        return fig
    
    def multi_scale_temporal_analysis(self,
                                    data: pd.DataFrame,
                                    time_col: str,
                                    value_col: str,
                                    scales: List[str] = ['microsecond', 'second', 'minute', 'hour'],
                                    title: str = "Multi-Scale Temporal Analysis") -> go.Figure:
        """
        Create multi-scale temporal analysis plots from microsecond to hour scale.
        
        Args:
            data: DataFrame with high-resolution temporal data
            time_col: Timestamp column
            value_col: Value to analyze
            scales: List of time scales to analyze
            title: Plot title
            
        Returns:
            Plotly figure with subplots for each scale
        """
        fig = make_subplots(
            rows=len(scales),
            cols=1,
            subplot_titles=[f'{scale.capitalize()} Scale Analysis' for scale in scales],
            vertical_spacing=0.08,
            shared_xaxes=False
        )
        
        # Define aggregation rules for each scale
        agg_rules = {
            'microsecond': '1us',
            'millisecond': '1ms',
            'second': '1s',
            'minute': '1min',
            'hour': '1h'
        }
        
        colors = px.colors.sequential.Viridis
        
        for idx, scale in enumerate(scales, 1):
            if scale in agg_rules:
                # Resample data at different scales
                resampled = data.set_index(time_col)[value_col].resample(agg_rules[scale]).agg(['mean', 'std', 'max'])
                
                # Add mean line
                fig.add_trace(
                    go.Scatter(
                        x=resampled.index,
                        y=resampled['mean'],
                        mode='lines',
                        name=f'{scale} mean',
                        line=dict(color=colors[idx*2], width=2),
                        showlegend=True
                    ),
                    row=idx, col=1
                )
                
                # Add confidence band
                fig.add_trace(
                    go.Scatter(
                        x=resampled.index,
                        y=resampled['mean'] + resampled['std'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=idx, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=resampled.index,
                        y=resampled['mean'] - resampled['std'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[idx*2])) + [0.2])}',
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=idx, col=1
                )
                
                # Add max points
                fig.add_trace(
                    go.Scatter(
                        x=resampled.index,
                        y=resampled['max'],
                        mode='markers',
                        name=f'{scale} peaks',
                        marker=dict(color=CUSTOM_COLORS['attack'], size=4),
                        showlegend=True
                    ),
                    row=idx, col=1
                )
        
        fig.update_layout(
            title={'text': title, 'font': {'size': 28, 'family': 'Arial Black'}},
            height=300 * len(scales),
            width=1200,
            paper_bgcolor=CUSTOM_COLORS['background'],
            plot_bgcolor='white',
            font=dict(family="Arial", size=12),
            showlegend=True,
            legend=dict(x=1.05, y=1)
        )
        
        fig.update_xaxes(title_text="Time", gridcolor=CUSTOM_COLORS['grid'])
        fig.update_yaxes(title_text=value_col, gridcolor=CUSTOM_COLORS['grid'])
        
        return fig
    
    def circular_iot_pattern_plot(self,
                                 data: pd.DataFrame,
                                 time_col: str,
                                 value_col: str,
                                 period: str = 'hour',
                                 title: str = "Cyclic IoT Activity Patterns") -> go.Figure:
        """
        Create circular/radial plots for visualizing cyclic IoT patterns.
        
        Args:
            data: DataFrame with temporal IoT data
            time_col: Timestamp column
            value_col: Value to visualize
            period: Cyclic period ('hour', 'day', 'week')
            title: Plot title
            
        Returns:
            Plotly polar plot figure
        """
        # Extract cyclic component based on period
        if period == 'hour':
            data['angle'] = pd.to_datetime(data[time_col]).dt.minute * 6  # 360/60
            labels = [f"{i:02d}" for i in range(0, 60, 5)]
        elif period == 'day':
            data['angle'] = pd.to_datetime(data[time_col]).dt.hour * 15  # 360/24
            labels = [f"{i:02d}:00" for i in range(0, 24, 2)]
        elif period == 'week':
            data['angle'] = pd.to_datetime(data[time_col]).dt.dayofweek * 51.43  # 360/7
            labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Aggregate data by angle
        agg_data = data.groupby('angle')[value_col].agg(['mean', 'std', 'count']).reset_index()
        
        # Create polar plot
        fig = go.Figure()
        
        # Add main trace
        fig.add_trace(go.Scatterpolar(
            r=agg_data['mean'],
            theta=agg_data['angle'],
            mode='lines+markers',
            name='Mean Activity',
            line=dict(color=CUSTOM_COLORS['accent'], width=3),
            marker=dict(size=8, color=CUSTOM_COLORS['accent']),
            fill='toself',
            fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(CUSTOM_COLORS["accent"])) + [0.2])}'
        ))
        
        # Add confidence bands
        fig.add_trace(go.Scatterpolar(
            r=agg_data['mean'] + agg_data['std'],
            theta=agg_data['angle'],
            mode='lines',
            name='Upper Bound (1 STD)',
            line=dict(color=CUSTOM_COLORS['attack'], width=2, dash='dash'),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=np.maximum(0, agg_data['mean'] - agg_data['std']),
            theta=agg_data['angle'],
            mode='lines',
            name='Lower Bound (1 STD)',
            line=dict(color=CUSTOM_COLORS['normal'], width=2, dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            title={'text': title, 'font': {'size': 26, 'family': 'Arial Black'}},
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor=CUSTOM_COLORS['grid'],
                    linecolor=CUSTOM_COLORS['grid']
                ),
                angularaxis=dict(
                    gridcolor=CUSTOM_COLORS['grid'],
                    linecolor=CUSTOM_COLORS['grid'],
                    rotation=90,
                    direction='clockwise'
                ),
                bgcolor='white'
            ),
            paper_bgcolor=CUSTOM_COLORS['background'],
            font=dict(family="Arial", size=14),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            width=800,
            height=800
        )
        
        return fig
    
    def attack_flow_sankey(self,
                          flow_data: pd.DataFrame,
                          source_col: str,
                          target_col: str,
                          value_col: str,
                          title: str = "Attack Flow Visualization") -> go.Figure:
        """
        Create Sankey diagram for attack flow visualization.
        
        Args:
            flow_data: DataFrame with flow information
            source_col: Source node column
            target_col: Target node column
            value_col: Flow value column
            title: Plot title
            
        Returns:
            Plotly Sankey diagram
        """
        # Create node labels
        all_nodes = pd.concat([flow_data[source_col], flow_data[target_col]]).unique()
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Map to indices
        source_indices = flow_data[source_col].map(node_dict)
        target_indices = flow_data[target_col].map(node_dict)
        
        # Create color scheme
        node_colors = []
        for node in all_nodes:
            if 'attack' in node.lower():
                node_colors.append(CUSTOM_COLORS['attack'])
            elif 'normal' in node.lower():
                node_colors.append(CUSTOM_COLORS['normal'])
            else:
                node_colors.append(CUSTOM_COLORS['transition'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(all_nodes),
                color=node_colors,
                hovertemplate='%{label}<br>Total flow: %{value}<extra></extra>'
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=flow_data[value_col],
                color=[f'rgba{tuple(list(px.colors.hex_to_rgb(CUSTOM_COLORS["transition"])) + [0.4])}']*len(flow_data),
                hovertemplate='%{source.label} → %{target.label}<br>Flow: %{value}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title={'text': title, 'font': {'size': 26, 'family': 'Arial Black'}},
            font=dict(size=14, family="Arial"),
            paper_bgcolor=CUSTOM_COLORS['background'],
            width=1200,
            height=600
        )
        
        return fig
    
    def interactive_iot_topology(self,
                               nodes: List[Dict],
                               edges: List[Dict],
                               attack_nodes: List[str] = None,
                               title: str = "IoT Network Topology with Attacks") -> go.Figure:
        """
        Create interactive network graph visualization for IoT topology with attack highlighting.
        
        Args:
            nodes: List of node dictionaries with 'id', 'label', 'type' keys
            edges: List of edge dictionaries with 'source', 'target', 'weight' keys
            attack_nodes: List of node IDs under attack
            title: Plot title
            
        Returns:
            Interactive plotly network graph
        """
        # Create networkx graph
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'], **node)
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1))
        
        # Use spring layout for positioning
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge traces
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Create node traces
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=[],
                color=[],
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            )
        )
        
        # Add node positions and properties
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
            # Node info
            node_info = G.nodes[node]
            node_trace['text'] += tuple([node_info.get('label', node)])
            
            # Color based on attack status
            if attack_nodes and node in attack_nodes:
                color = CUSTOM_COLORS['attack']
                size = 30
            else:
                color = len(list(G.neighbors(node)))
                size = 20
            
            node_trace['marker']['color'] += tuple([color])
            node_trace['marker']['size'] += tuple([size])
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title={'text': title, 'font': {'size': 26, 'family': 'Arial Black'}},
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor=CUSTOM_COLORS['background'],
            plot_bgcolor='white',
            width=1000,
            height=800
        )
        
        # Add annotations for attack nodes
        if attack_nodes:
            annotations = []
            for node in attack_nodes:
                if node in pos:
                    x, y = pos[node]
                    annotations.append(dict(
                        x=x,
                        y=y,
                        text="⚠️",
                        showarrow=False,
                        font=dict(size=20)
                    ))
            fig.update_layout(annotations=annotations)
        
        return fig
    
    def distribution_violin_plots(self,
                                data: pd.DataFrame,
                                feature_cols: List[str],
                                category_col: str,
                                title: str = "Feature Distribution During Attacks") -> plt.Figure:
        """
        Create violin plots showing distribution changes during attacks.
        
        Args:
            data: DataFrame with feature data
            feature_cols: List of feature columns to visualize
            category_col: Column distinguishing normal/attack states
            title: Plot title
            
        Returns:
            Matplotlib figure with violin plots
        """
        n_features = len(feature_cols)
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 8))
        
        if n_features == 1:
            axes = [axes]
        
        for idx, (ax, feature) in enumerate(zip(axes, feature_cols)):
            # Create violin plot
            parts = ax.violinplot(
                [data[data[category_col] == cat][feature].dropna() 
                 for cat in data[category_col].unique()],
                positions=range(len(data[category_col].unique())),
                showmeans=True,
                showextrema=True,
                showmedians=True
            )
            
            # Customize colors
            colors = [CUSTOM_COLORS['normal'], CUSTOM_COLORS['attack']]
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Style the plot
            ax.set_title(f'{feature} Distribution', fontsize=16, weight='bold')
            ax.set_xticks(range(len(data[category_col].unique())))
            ax.set_xticklabels(data[category_col].unique())
            ax.set_ylabel('Value', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add statistical annotations
            for i, cat in enumerate(data[category_col].unique()):
                data_cat = data[data[category_col] == cat][feature].dropna()
                mean_val = data_cat.mean()
                std_val = data_cat.std()
                ax.text(i, ax.get_ylim()[1]*0.95, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                       ha='center', va='top', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=20, weight='bold')
        fig.tight_layout()
        
        return fig
    
    def temporal_ridge_plot(self,
                          data: pd.DataFrame,
                          time_col: str,
                          value_col: str,
                          n_ridges: int = 20,
                          title: str = "Temporal Density Evolution") -> plt.Figure:
        """
        Create ridge plots for temporal density evolution.
        
        Args:
            data: DataFrame with temporal data
            time_col: Time column name
            value_col: Value column to show density
            n_ridges: Number of time slices
            title: Plot title
            
        Returns:
            Matplotlib figure with ridge plot
        """
        # Create time bins
        data['time_bin'] = pd.cut(data[time_col], bins=n_ridges, labels=False)
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Color palette
        colors = sns.color_palette("coolwarm", n_ridges)
        
        # Create ridge plot
        for i in range(n_ridges):
            # Get data for this time slice
            subset = data[data['time_bin'] == i][value_col].dropna()
            if len(subset) > 1:
                # Create KDE
                density = stats.gaussian_kde(subset)
                xs = np.linspace(subset.min(), subset.max(), 200)
                ys = density(xs)
                
                # Normalize and offset
                ys = ys / ys.max() * 0.9
                ys += i
                
                # Plot
                ax.fill_between(xs, i, ys, alpha=0.7, color=colors[i], zorder=n_ridges-i)
                ax.plot(xs, ys, color='white', linewidth=1.5, zorder=n_ridges-i+0.5)
        
        # Styling
        ax.set_xlim(data[value_col].min(), data[value_col].max())
        ax.set_ylim(-0.5, n_ridges)
        ax.set_xlabel(value_col, fontsize=16, weight='bold')
        ax.set_ylabel('Time Evolution →', fontsize=16, weight='bold')
        ax.set_title(title, fontsize=20, weight='bold', pad=20)
        
        # Clean up axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        
        # Add time labels
        time_range = data[time_col].max() - data[time_col].min()
        time_labels = []
        for i in range(0, n_ridges, max(1, n_ridges//5)):
            time_point = data[time_col].min() + (i/n_ridges) * time_range
            time_labels.append((i, str(time_point)[:10]))
        
        for pos, label in time_labels:
            ax.text(-0.05, pos, label, transform=ax.get_yaxis_transform(), 
                   ha='right', va='center', fontsize=10)
        
        fig.tight_layout()
        return fig
    
    def create_interactive_dashboard(self,
                                   data: pd.DataFrame,
                                   features: List[str],
                                   time_col: str,
                                   attack_col: str,
                                   title: str = "IoT Attack Detection Dashboard") -> go.Figure:
        """
        Create a comprehensive interactive dashboard with multiple visualizations.
        
        Args:
            data: Complete dataset
            features: List of feature columns
            time_col: Time column name
            attack_col: Attack indicator column
            title: Dashboard title
            
        Returns:
            Plotly dashboard figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Time Series Overview', 'Feature Correlation',
                          'Attack Distribution', 'Anomaly Scores',
                          'Network Activity', 'Detection Performance'),
            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
                   [{'type': 'pie'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Time series overview
        for idx, feature in enumerate(features[:3]):
            fig.add_trace(
                go.Scatter(
                    x=data[time_col],
                    y=data[feature],
                    mode='lines',
                    name=feature,
                    line=dict(width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # 2. Feature correlation heatmap
        corr_matrix = data[features].corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                showscale=True
            ),
            row=1, col=2
        )
        
        # 3. Attack distribution pie chart
        attack_counts = data[attack_col].value_counts()
        fig.add_trace(
            go.Pie(
                labels=attack_counts.index,
                values=attack_counts.values,
                hole=0.3,
                marker=dict(colors=[CUSTOM_COLORS['normal'], CUSTOM_COLORS['attack']]),
                textinfo='label+percent'
            ),
            row=2, col=1
        )
        
        # 4. Anomaly scores scatter
        if 'anomaly_score' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data[time_col],
                    y=data['anomaly_score'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=data[attack_col].map({0: CUSTOM_COLORS['normal'], 1: CUSTOM_COLORS['attack']}),
                        opacity=0.6
                    ),
                    name='Anomaly Scores'
                ),
                row=2, col=2
            )
        
        # 5. Network activity over time
        time_grouped = data.groupby([pd.Grouper(key=time_col, freq='1h'), attack_col]).size().reset_index(name='count')
        for attack_type in time_grouped[attack_col].unique():
            subset = time_grouped[time_grouped[attack_col] == attack_type]
            fig.add_trace(
                go.Scatter(
                    x=subset[time_col],
                    y=subset['count'],
                    mode='lines+markers',
                    name=f'Activity - {"Attack" if attack_type == 1 else "Normal"}',
                    line=dict(width=3)
                ),
                row=3, col=1
            )
        
        # 6. Detection performance metrics
        if 'prediction' in data.columns:
            metrics = {
                'True Positive': ((data[attack_col] == 1) & (data['prediction'] == 1)).sum(),
                'True Negative': ((data[attack_col] == 0) & (data['prediction'] == 0)).sum(),
                'False Positive': ((data[attack_col] == 0) & (data['prediction'] == 1)).sum(),
                'False Negative': ((data[attack_col] == 1) & (data['prediction'] == 0)).sum()
            }
            fig.add_trace(
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker=dict(color=[CUSTOM_COLORS['normal'], CUSTOM_COLORS['normal'], 
                                     CUSTOM_COLORS['attack'], CUSTOM_COLORS['attack']])
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title={'text': title, 'font': {'size': 30, 'family': 'Arial Black'}},
            height=1200,
            width=1600,
            showlegend=True,
            paper_bgcolor=CUSTOM_COLORS['background'],
            font=dict(family="Arial", size=12),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial"
            )
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor=CUSTOM_COLORS['grid'])
        fig.update_yaxes(showgrid=True, gridcolor=CUSTOM_COLORS['grid'])
        
        return fig


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 10000
    
    # Generate temporal data
    time_range = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')
    
    # Create synthetic IoT data
    demo_data = pd.DataFrame({
        'timestamp': time_range,
        'cpu_usage': np.random.normal(30, 10, n_samples) + np.sin(np.arange(n_samples) * 0.01) * 20,
        'memory_usage': np.random.normal(50, 15, n_samples),
        'network_traffic': np.random.exponential(100, n_samples),
        'temperature': np.random.normal(25, 2, n_samples) + np.cos(np.arange(n_samples) * 0.005) * 5,
        'packet_size': np.random.gamma(2, 50, n_samples),
        'response_time': np.random.lognormal(3, 0.5, n_samples)
    })
    
    # Add attack labels (simulate attacks)
    attack_periods = [(1000, 1500), (3000, 3200), (7000, 7500)]
    demo_data['is_attack'] = 0
    for start, end in attack_periods:
        demo_data.loc[start:end, 'is_attack'] = 1
        # Modify features during attacks
        demo_data.loc[start:end, 'cpu_usage'] *= 2.5
        demo_data.loc[start:end, 'network_traffic'] *= 5
        demo_data.loc[start:end, 'response_time'] *= 3
    
    # Calculate anomaly scores
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(contamination=0.1, random_state=42)
    feature_cols = ['cpu_usage', 'memory_usage', 'network_traffic', 'temperature', 'packet_size', 'response_time']
    anomaly_scores = clf.decision_function(demo_data[feature_cols])
    demo_data['anomaly_score'] = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    # Initialize visualizer
    viz = TemporalDeceptionVisualizer(style='research')
    
    # Example 1: 3D Temporal Heatmap
    print("Creating 3D temporal heatmap...")
    fig_3d = viz.create_3d_temporal_heatmap(
        demo_data.iloc[:1000], 
        'timestamp', 
        ['cpu_usage', 'memory_usage'], 
        'anomaly_score',
        "3D Temporal Anomaly Heatmap - CPU vs Memory Usage"
    )
    # fig_3d.show()
    
    # Example 2: Animated Attack Evolution
    print("Creating animated attack evolution...")
    fig_animated = viz.animated_attack_evolution(
        demo_data.iloc[:500],
        'timestamp',
        'cpu_usage',
        'network_traffic',
        'is_attack',
        "Attack Evolution: CPU vs Network Traffic"
    )
    # fig_animated.show()
    
    # Example 3: Multi-scale Analysis
    print("Creating multi-scale temporal analysis...")
    fig_multiscale = viz.multi_scale_temporal_analysis(
        demo_data.iloc[:1000],
        'timestamp',
        'cpu_usage',
        ['second', 'minute', 'hour'],
        "Multi-Scale CPU Usage Analysis"
    )
    # fig_multiscale.show()
    
    # Example 4: Circular Pattern Plot
    print("Creating circular IoT pattern plot...")
    fig_circular = viz.circular_iot_pattern_plot(
        demo_data,
        'timestamp',
        'temperature',
        'day',
        "Daily Temperature Patterns in IoT Devices"
    )
    # fig_circular.show()
    
    # Example 5: Attack Flow Sankey
    print("Creating attack flow Sankey diagram...")
    flow_data = pd.DataFrame({
        'source': ['Normal State', 'Normal State', 'Scanning', 'Scanning', 'Exploitation'],
        'target': ['Scanning', 'Exploitation', 'Exploitation', 'Data Exfiltration', 'Data Exfiltration'],
        'value': [150, 50, 100, 80, 120]
    })
    fig_sankey = viz.attack_flow_sankey(
        flow_data,
        'source',
        'target',
        'value',
        "IoT Attack Flow Progression"
    )
    # fig_sankey.show()
    
    # Example 6: Interactive Network Topology
    print("Creating interactive IoT network topology...")
    nodes = [
        {'id': 'gateway', 'label': 'Gateway', 'type': 'router'},
        {'id': 'sensor1', 'label': 'Temp Sensor', 'type': 'sensor'},
        {'id': 'sensor2', 'label': 'Motion Sensor', 'type': 'sensor'},
        {'id': 'actuator1', 'label': 'Smart Lock', 'type': 'actuator'},
        {'id': 'server', 'label': 'Cloud Server', 'type': 'server'}
    ]
    edges = [
        {'source': 'gateway', 'target': 'sensor1', 'weight': 10},
        {'source': 'gateway', 'target': 'sensor2', 'weight': 8},
        {'source': 'gateway', 'target': 'actuator1', 'weight': 15},
        {'source': 'gateway', 'target': 'server', 'weight': 50}
    ]
    fig_topology = viz.interactive_iot_topology(
        nodes,
        edges,
        ['sensor2', 'actuator1'],
        "IoT Network Under Attack"
    )
    # fig_topology.show()
    
    # Example 7: Distribution Violin Plots
    print("Creating distribution violin plots...")
    fig_violin = viz.distribution_violin_plots(
        demo_data,
        ['cpu_usage', 'network_traffic', 'response_time'],
        'is_attack',
        "Feature Distributions: Normal vs Attack States"
    )
    # plt.show()
    
    # Example 8: Temporal Ridge Plot
    print("Creating temporal ridge plot...")
    fig_ridge = viz.temporal_ridge_plot(
        demo_data,
        'timestamp',
        'network_traffic',
        20,
        "Network Traffic Density Evolution Over Time"
    )
    # plt.show()
    
    # Example 9: Interactive Dashboard
    print("Creating interactive dashboard...")
    demo_data['prediction'] = (demo_data['anomaly_score'] > 0.7).astype(int)
    fig_dashboard = viz.create_interactive_dashboard(
        demo_data.iloc[:2000],
        feature_cols,
        'timestamp',
        'is_attack',
        "Comprehensive IoT Security Monitoring Dashboard"
    )
    # fig_dashboard.show()
    
    print("\nAll visualizations created successfully!")
    print("\nTo use this library in your research:")
    print("1. Import the TemporalDeceptionVisualizer class")
    print("2. Initialize with your preferred style")
    print("3. Call the appropriate visualization method with your data")
    print("4. Save figures using fig.write_html() for Plotly or plt.savefig() for Matplotlib")