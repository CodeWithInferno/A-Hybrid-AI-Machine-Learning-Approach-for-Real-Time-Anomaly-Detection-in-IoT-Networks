# Advanced Visualization Library for Temporal Deception Detection

A sophisticated, research-grade visualization toolkit specifically designed for IoT security research and temporal deception detection analysis. This library provides publication-quality plots with interactive features for comprehensive analysis of IoT attack patterns.

## Features

### 🎯 Core Visualization Types

1. **3D Temporal Heatmaps** - Advanced 3D surface plots with Plotly for multi-dimensional temporal analysis
2. **Animated Attack Evolution** - Real-time attack progression tracking with smooth animations
3. **Multi-Scale Temporal Analysis** - Simultaneous analysis from microseconds to hours
4. **Circular/Radial Pattern Plots** - Specialized for cyclic IoT behavior patterns
5. **Sankey Flow Diagrams** - Attack progression and cyber kill chain visualization
6. **Interactive Network Topology** - IoT device networks with attack highlighting
7. **Distribution Violin Plots** - Statistical distribution changes during attacks
8. **Temporal Ridge Plots** - Density evolution over time periods
9. **Comprehensive Dashboards** - Multi-panel interactive analysis interfaces

### 🎨 Design Philosophy

- **Research-Grade Quality**: Publication-ready styling with high DPI output
- **Interactive Elements**: Hover details, zoom, pan, and animation controls
- **Custom Color Schemes**: Scientifically-informed color palettes for clarity
- **Performance Optimized**: Efficient rendering for large temporal datasets
- **Modular Architecture**: Easy to extend and customize for specific research needs

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd New_Approach/visualization

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from top_tier_visualizations import TemporalDeceptionVisualizer
import pandas as pd
import numpy as np

# Initialize the visualizer
viz = TemporalDeceptionVisualizer(style='research')

# Create sample data
data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
    'cpu_usage': np.random.normal(30, 10, 1000),
    'network_traffic': np.random.exponential(100, 1000),
    'anomaly_score': np.random.beta(2, 8, 1000),
    'is_attack': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
})

# Create 3D temporal heatmap
fig = viz.create_3d_temporal_heatmap(
    data, 'timestamp', ['cpu_usage'], 'anomaly_score',
    title="3D Attack Detection Analysis"
)
fig.show()

# Create animated attack evolution
fig_anim = viz.animated_attack_evolution(
    data, 'timestamp', 'cpu_usage', 'network_traffic', 'is_attack',
    title="Real-time Attack Evolution"
)
fig_anim.show()
```

## Comprehensive Example

Run the complete demonstration:

```bash
python example_usage.py
```

This will generate:
- 8 interactive HTML visualizations
- 2 high-resolution PNG plots
- Sample data with realistic IoT attack patterns

## API Reference

### TemporalDeceptionVisualizer Class

#### Core Methods

**`create_3d_temporal_heatmap(data, time_col, feature_cols, value_col, title)`**
- Creates advanced 3D surface plots for temporal analysis
- Supports multi-feature visualization with surface contours
- Interactive 3D rotation and zoom capabilities

**`animated_attack_evolution(data, time_col, x_col, y_col, category_col, title)`**
- Generates animated scatter plots showing attack progression
- Customizable playback speed and animation controls
- Color-coded by attack/normal states

**`multi_scale_temporal_analysis(data, time_col, value_col, scales, title)`**
- Simultaneous analysis across multiple time scales
- Aggregation from microseconds to hours
- Statistical confidence bands and peak detection

**`circular_iot_pattern_plot(data, time_col, value_col, period, title)`**
- Polar coordinate plots for cyclic patterns
- Supports hourly, daily, and weekly cycles
- Statistical confidence visualization

**`attack_flow_sankey(flow_data, source_col, target_col, value_col, title)`**
- Sankey diagrams for attack progression flows
- Customizable node colors and flow thickness
- Interactive hover information

**`interactive_iot_topology(nodes, edges, attack_nodes, title)`**
- Network graph visualization with attack highlighting
- Spring-layout positioning with customizable physics
- Interactive node information and zoom capabilities

**`distribution_violin_plots(data, feature_cols, category_col, title)`**
- Statistical distribution comparison plots
- Violin plots with overlaid statistics
- Multi-feature support with subplots

**`temporal_ridge_plot(data, time_col, value_col, n_ridges, title)`**
- Ridge plots showing density evolution over time
- Kernel density estimation with time slicing
- Gradient coloring for temporal progression

**`create_interactive_dashboard(data, features, time_col, attack_col, title)`**
- Comprehensive multi-panel dashboard
- Real-time performance metrics
- Correlation analysis and attack distribution

#### Styling Options

```python
# Available styles
viz = TemporalDeceptionVisualizer(style='research')  # Publication quality
viz = TemporalDeceptionVisualizer(style='dark')      # Dark theme
viz = TemporalDeceptionVisualizer(style='minimal')   # Clean minimal
```

## Data Format Requirements

The library expects pandas DataFrames with the following column types:

- **Timestamp columns**: pandas datetime objects
- **Feature columns**: numeric (int/float)
- **Category columns**: categorical or binary (0/1)
- **Value columns**: numeric measurements or scores

### Sample Data Structure

```python
data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=N, freq='30s'),
    'cpu_usage': float,        # System metrics
    'memory_usage': float,
    'network_traffic': float,
    'temperature': float,      # Environmental sensors
    'response_time': float,    # Performance metrics
    'is_attack': int,          # Binary attack indicator
    'anomaly_score': float,    # Computed anomaly scores
    'device_id': str,          # Device identifiers
    'attack_type': str         # Attack categorization
})
```

## Research Applications

This library is specifically designed for:

- **IoT Security Research**: Temporal attack pattern analysis
- **Anomaly Detection Studies**: Multi-scale temporal anomaly visualization
- **Network Security Analysis**: Attack flow and progression tracking
- **Academic Publications**: High-quality figures with publication standards
- **Real-time Monitoring**: Interactive dashboards for security operations
- **Comparative Studies**: Side-by-side attack vs. normal state analysis

## Performance Considerations

- **Large Datasets**: Use data sampling for datasets >50k points in interactive plots
- **3D Visualizations**: Limit to <5k points for smooth rotation
- **Animations**: Recommended <1k frames for optimal playback
- **Dashboard**: Use <10k points for responsive interaction

## Customization

### Custom Color Schemes

```python
# Modify the global color scheme
from top_tier_visualizations import CUSTOM_COLORS
CUSTOM_COLORS['attack'] = '#FF6B6B'  # Custom attack color
CUSTOM_COLORS['normal'] = '#4ECDC4'  # Custom normal color
```

### Export Options

```python
# For Plotly figures (HTML, PNG, PDF, SVG)
fig.write_html("output.html")
fig.write_image("output.png", width=1200, height=800, scale=2)

# For Matplotlib figures
fig.savefig("output.png", dpi=300, bbox_inches='tight')
fig.savefig("output.pdf", format='pdf', bbox_inches='tight')
```

## Contributing

This library is designed for research use. When extending functionality:

1. Maintain publication-quality styling standards
2. Include comprehensive docstrings with examples
3. Add type hints for all parameters
4. Ensure interactive elements are responsive
5. Test with various data scales and formats

## Dependencies

- `plotly >= 5.17.0` - Interactive plotting
- `matplotlib >= 3.7.0` - Static plotting
- `seaborn >= 0.12.0` - Statistical visualization
- `pandas >= 2.0.0` - Data manipulation
- `numpy >= 1.24.0` - Numerical computing
- `networkx >= 3.1` - Network analysis
- `scikit-learn >= 1.3.0` - Machine learning utilities

## License

This visualization library is developed for academic and research purposes. Please cite appropriately when using in publications.

## Support

For research collaboration or technical questions, please open an issue or contact the development team.

---

**Research-Grade IoT Security Visualization** | Built for temporal deception detection and IoT attack analysis