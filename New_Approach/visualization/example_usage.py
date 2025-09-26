"""
Example Usage Script for Advanced IoT Visualization Library
===========================================================

This script demonstrates how to use the TemporalDeceptionVisualizer class
for creating publication-quality visualizations for IoT security research.

Run this script to generate sample visualizations and save them as HTML files.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import the visualization class
from top_tier_visualizations import TemporalDeceptionVisualizer


def generate_realistic_iot_data(n_samples=20000, attack_ratio=0.15):
    """
    Generate realistic IoT dataset with temporal patterns and attacks.
    
    Args:
        n_samples (int): Number of data points to generate
        attack_ratio (float): Ratio of attack samples
        
    Returns:
        pd.DataFrame: Generated IoT data with attacks
    """
    print(f"Generating {n_samples} IoT data points with {attack_ratio:.1%} attacks...")
    
    np.random.seed(42)
    
    # Create temporal index
    start_time = datetime(2024, 1, 1)
    time_range = pd.date_range(start=start_time, periods=n_samples, freq='30s')
    
    # Generate base features with temporal patterns
    t = np.arange(n_samples)
    
    # CPU usage with daily and weekly patterns
    daily_pattern = 20 * np.sin(2 * np.pi * t / (24 * 120))  # 24 hours in 30s intervals
    weekly_pattern = 10 * np.cos(2 * np.pi * t / (7 * 24 * 120))  # 7 days
    cpu_base = 35 + daily_pattern + weekly_pattern + np.random.normal(0, 8, n_samples)
    
    # Memory usage with gradual increase over time
    memory_trend = 0.001 * t  # Gradual memory increase
    memory_base = 45 + memory_trend + np.random.normal(0, 12, n_samples)
    
    # Network traffic with bursty patterns
    network_base = np.random.exponential(150, n_samples) * (1 + 0.3 * np.sin(2 * np.pi * t / 100))
    
    # Temperature with environmental patterns
    temp_daily = 5 * np.sin(2 * np.pi * t / (24 * 120) - np.pi/2)  # Peak at noon
    temp_base = 22 + temp_daily + np.random.normal(0, 1.5, n_samples)
    
    # Response time (log-normal distribution)
    response_base = np.random.lognormal(mean=3.5, sigma=0.8, size=n_samples)
    
    # Packet size distribution
    packet_base = np.random.gamma(shape=2, scale=200, size=n_samples)
    
    # Connection count following Poisson distribution
    connections_base = np.random.poisson(lam=5, size=n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': time_range,
        'cpu_usage': np.clip(cpu_base, 0, 100),
        'memory_usage': np.clip(memory_base, 0, 100),
        'network_traffic': np.clip(network_base, 0, 10000),
        'temperature': temp_base,
        'response_time': response_base,
        'packet_size': packet_base,
        'connections': connections_base,
        'is_attack': 0
    })
    
    # Inject realistic attacks
    n_attacks = int(n_samples * attack_ratio)
    attack_indices = []
    
    # Create different attack types with different durations
    attack_types = [
        ('ddos', 300, 2.5),      # DDoS: 300 samples duration, 2.5x multiplier
        ('scan', 100, 1.8),      # Port scan: shorter duration
        ('botnet', 500, 3.0),    # Botnet: longer duration, higher impact
        ('malware', 200, 2.2)    # Malware: moderate duration
    ]
    
    current_attacks = 0
    while current_attacks < n_attacks:
        # Choose random attack type
        attack_type, duration, multiplier = attack_types[np.random.randint(len(attack_types))]
        
        # Choose random start position (ensure no overlap)
        max_start = n_samples - duration - 1
        if max_start <= 0:
            break
            
        start_idx = np.random.randint(0, max_start)
        
        # Check for overlap with existing attacks
        if any(abs(start_idx - existing) < duration for existing in attack_indices):
            continue
            
        attack_indices.append(start_idx)
        end_idx = min(start_idx + duration, n_samples)
        
        # Mark as attack
        data.loc[start_idx:end_idx, 'is_attack'] = 1
        
        # Modify features based on attack type
        if attack_type == 'ddos':
            data.loc[start_idx:end_idx, 'network_traffic'] *= multiplier
            data.loc[start_idx:end_idx, 'connections'] *= 4
            data.loc[start_idx:end_idx, 'response_time'] *= 2.5
        elif attack_type == 'scan':
            data.loc[start_idx:end_idx, 'connections'] *= 10
            data.loc[start_idx:end_idx, 'packet_size'] *= 0.3  # Small packets
        elif attack_type == 'botnet':
            data.loc[start_idx:end_idx, 'cpu_usage'] *= multiplier
            data.loc[start_idx:end_idx, 'memory_usage'] *= 1.5
            data.loc[start_idx:end_idx, 'network_traffic'] *= 2
        elif attack_type == 'malware':
            data.loc[start_idx:end_idx, 'cpu_usage'] *= multiplier
            data.loc[start_idx:end_idx, 'response_time'] *= 3
        
        current_attacks += duration
    
    # Calculate anomaly scores using Isolation Forest
    print("Calculating anomaly scores...")
    feature_cols = ['cpu_usage', 'memory_usage', 'network_traffic', 'temperature', 
                   'response_time', 'packet_size', 'connections']
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data[feature_cols])
    
    # Detect anomalies
    isolation_forest = IsolationForest(contamination=attack_ratio + 0.05, random_state=42)
    anomaly_scores = isolation_forest.decision_function(features_scaled)
    
    # Normalize anomaly scores to 0-1 range
    data['anomaly_score'] = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    # Generate predictions based on anomaly scores
    threshold = np.percentile(data['anomaly_score'], (1-attack_ratio-0.05)*100)
    data['prediction'] = (data['anomaly_score'] > threshold).astype(int)
    
    print(f"Dataset created: {len(data)} samples, {data['is_attack'].sum()} attacks ({data['is_attack'].mean():.2%})")
    return data


def create_network_topology_data():
    """Create sample network topology data for visualization."""
    nodes = [
        {'id': 'gateway', 'label': 'IoT Gateway', 'type': 'gateway'},
        {'id': 'temp1', 'label': 'Temperature Sensor 1', 'type': 'sensor'},
        {'id': 'temp2', 'label': 'Temperature Sensor 2', 'type': 'sensor'},
        {'id': 'motion1', 'label': 'Motion Detector', 'type': 'sensor'},
        {'id': 'camera1', 'label': 'Security Camera', 'type': 'camera'},
        {'id': 'lock1', 'label': 'Smart Lock', 'type': 'actuator'},
        {'id': 'thermostat', 'label': 'Smart Thermostat', 'type': 'actuator'},
        {'id': 'server', 'label': 'Cloud Server', 'type': 'server'},
        {'id': 'mobile', 'label': 'Mobile App', 'type': 'client'},
        {'id': 'hub', 'label': 'Smart Hub', 'type': 'hub'}
    ]
    
    edges = [
        {'source': 'gateway', 'target': 'temp1', 'weight': 15},
        {'source': 'gateway', 'target': 'temp2', 'weight': 12},
        {'source': 'gateway', 'target': 'motion1', 'weight': 20},
        {'source': 'gateway', 'target': 'camera1', 'weight': 100},
        {'source': 'gateway', 'target': 'lock1', 'weight': 25},
        {'source': 'gateway', 'target': 'thermostat', 'weight': 30},
        {'source': 'gateway', 'target': 'hub', 'weight': 50},
        {'source': 'hub', 'target': 'server', 'weight': 200},
        {'source': 'server', 'target': 'mobile', 'weight': 150},
        {'source': 'hub', 'target': 'mobile', 'weight': 80}
    ]
    
    # Simulate attacks on specific nodes
    attack_nodes = ['camera1', 'lock1', 'temp2']
    
    return nodes, edges, attack_nodes


def create_attack_flow_data():
    """Create attack flow data for Sankey diagram."""
    flow_data = pd.DataFrame({
        'source': [
            'Normal Operation', 'Normal Operation', 'Normal Operation',
            'Reconnaissance', 'Reconnaissance', 'Reconnaissance',
            'Initial Access', 'Initial Access', 'Initial Access',
            'Persistence', 'Privilege Escalation', 'Lateral Movement',
            'Data Collection', 'Command & Control'
        ],
        'target': [
            'Reconnaissance', 'Initial Access', 'No Attack',
            'Initial Access', 'Failed Access', 'Normal Operation',
            'Persistence', 'Failed Access', 'Normal Operation',
            'Privilege Escalation', 'Lateral Movement', 'Data Collection',
            'Data Exfiltration', 'Data Exfiltration'
        ],
        'value': [
            200, 50, 8500,  # From Normal Operation
            150, 75, 25,    # From Reconnaissance
            120, 30, 0,     # From Initial Access
            100, 80, 60,    # Single source flows
            90, 70          # Final stages
        ]
    })
    
    return flow_data


def main():
    """Main function to demonstrate all visualization capabilities."""
    print("=== Advanced IoT Visualization Library Demo ===\n")
    
    # Generate realistic IoT data
    data = generate_realistic_iot_data(n_samples=15000, attack_ratio=0.12)
    
    # Initialize the visualizer
    print("\nInitializing visualization library...")
    viz = TemporalDeceptionVisualizer(style='research')
    
    # Feature columns for analysis
    feature_cols = ['cpu_usage', 'memory_usage', 'network_traffic', 'temperature', 
                   'response_time', 'packet_size', 'connections']
    
    print("\n--- Creating Advanced Visualizations ---\n")
    
    # 1. 3D Temporal Heatmap
    print("1. Creating 3D temporal heatmap...")
    fig_3d = viz.create_3d_temporal_heatmap(
        data.iloc[:2000],  # Use subset for better performance
        'timestamp',
        ['cpu_usage', 'memory_usage'],
        'anomaly_score',
        "3D Temporal Anomaly Detection: CPU vs Memory Usage"
    )
    fig_3d.write_html("3d_temporal_heatmap.html")
    print("   Saved: 3d_temporal_heatmap.html")
    
    # 2. Animated Attack Evolution
    print("2. Creating animated attack evolution...")
    attack_subset = data[data['is_attack'] == 1].iloc[:500]  # Focus on attack periods
    if len(attack_subset) > 0:
        fig_animated = viz.animated_attack_evolution(
            attack_subset,
            'timestamp',
            'cpu_usage',
            'network_traffic',
            'is_attack',
            "Real-time Attack Evolution: CPU vs Network Traffic"
        )
        fig_animated.write_html("animated_attack_evolution.html")
        print("   Saved: animated_attack_evolution.html")
    
    # 3. Multi-scale Temporal Analysis
    print("3. Creating multi-scale temporal analysis...")
    fig_multiscale = viz.multi_scale_temporal_analysis(
        data.iloc[:5000],
        'timestamp',
        'cpu_usage',
        ['second', 'minute', 'hour'],
        "Multi-Scale CPU Usage Analysis Across Time Domains"
    )
    fig_multiscale.write_html("multiscale_analysis.html")
    print("   Saved: multiscale_analysis.html")
    
    # 4. Circular IoT Pattern Plot
    print("4. Creating circular IoT pattern plot...")
    fig_circular = viz.circular_iot_pattern_plot(
        data,
        'timestamp',
        'temperature',
        'day',
        "Daily Temperature Patterns in IoT Environment"
    )
    fig_circular.write_html("circular_patterns.html")
    print("   Saved: circular_patterns.html")
    
    # 5. Attack Flow Sankey
    print("5. Creating attack flow Sankey diagram...")
    flow_data = create_attack_flow_data()
    fig_sankey = viz.attack_flow_sankey(
        flow_data,
        'source',
        'target',
        'value',
        "Cyber Kill Chain: IoT Attack Flow Analysis"
    )
    fig_sankey.write_html("attack_flow_sankey.html")
    print("   Saved: attack_flow_sankey.html")
    
    # 6. Interactive Network Topology
    print("6. Creating interactive network topology...")
    nodes, edges, attack_nodes = create_network_topology_data()
    fig_topology = viz.interactive_iot_topology(
        nodes,
        edges,
        attack_nodes,
        "IoT Network Topology Under Cyber Attack"
    )
    fig_topology.write_html("network_topology.html")
    print("   Saved: network_topology.html")
    
    # 7. Distribution Violin Plots
    print("7. Creating distribution violin plots...")
    fig_violin = viz.distribution_violin_plots(
        data.sample(5000),  # Sample for better visualization
        ['cpu_usage', 'network_traffic', 'response_time', 'connections'],
        'is_attack',
        "Feature Distribution Analysis: Normal vs Attack States"
    )
    fig_violin.savefig("violin_distributions.png", dpi=300, bbox_inches='tight')
    print("   Saved: violin_distributions.png")
    
    # 8. Temporal Ridge Plot
    print("8. Creating temporal ridge plot...")
    fig_ridge = viz.temporal_ridge_plot(
        data,
        'timestamp',
        'network_traffic',
        25,
        "Network Traffic Density Evolution: Temporal Ridge Analysis"
    )
    fig_ridge.savefig("temporal_ridge.png", dpi=300, bbox_inches='tight')
    print("   Saved: temporal_ridge.png")
    
    # 9. Comprehensive Interactive Dashboard
    print("9. Creating comprehensive dashboard...")
    dashboard_data = data.iloc[:8000]  # Use subset for better performance
    fig_dashboard = viz.create_interactive_dashboard(
        dashboard_data,
        feature_cols[:4],  # Use main features
        'timestamp',
        'is_attack',
        "IoT Security Monitoring: Comprehensive Analysis Dashboard"
    )
    fig_dashboard.write_html("comprehensive_dashboard.html")
    print("   Saved: comprehensive_dashboard.html")
    
    # Create additional specialized plots
    print("\n--- Creating Specialized Research Plots ---\n")
    
    # 10. Attack Timeline Visualization
    print("10. Creating attack timeline...")
    attack_timeline = data[data['is_attack'] == 1].groupby(
        pd.Grouper(key='timestamp', freq='1h')
    ).size().reset_index(name='attack_count')
    
    fig_timeline = viz.create_3d_temporal_heatmap(
        data.iloc[:3000],
        'timestamp',
        ['response_time', 'connections'],
        'anomaly_score',
        "Attack Detection Timeline: Response Time vs Connections"
    )
    fig_timeline.write_html("attack_timeline_3d.html")
    print("   Saved: attack_timeline_3d.html")
    
    print(f"\n=== Visualization Demo Complete ===")
    print(f"Generated {10} different visualization types")
    print(f"Dataset statistics:")
    print(f"  - Total samples: {len(data):,}")
    print(f"  - Attack samples: {data['is_attack'].sum():,} ({data['is_attack'].mean():.2%})")
    print(f"  - Time range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"  - Features analyzed: {len(feature_cols)}")
    
    print(f"\nFiles created:")
    files_created = [
        "3d_temporal_heatmap.html",
        "animated_attack_evolution.html", 
        "multiscale_analysis.html",
        "circular_patterns.html",
        "attack_flow_sankey.html",
        "network_topology.html",
        "violin_distributions.png",
        "temporal_ridge.png",
        "comprehensive_dashboard.html",
        "attack_timeline_3d.html"
    ]
    
    for file in files_created:
        print(f"  - {file}")
    
    print(f"\nOpen the HTML files in your browser to explore the interactive visualizations!")
    print(f"The PNG files can be used directly in research publications.")


if __name__ == "__main__":
    main()