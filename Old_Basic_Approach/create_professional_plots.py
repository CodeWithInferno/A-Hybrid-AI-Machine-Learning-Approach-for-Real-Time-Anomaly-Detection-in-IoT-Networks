"""
Professional Visualization Generator for IoT Botnet Detection Research
Creates clean, publication-ready plots for research report
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set professional style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'DejaVu Sans'
})

def create_performance_comparison():
    """Create professional model performance comparison chart"""
    # Performance data from experiment results
    models = ['Statistical-Only\n(Isolation Forest)', 'LSTM-Only\n(Deep Autoencoder)', 'Hybrid\n(Fusion Model)']
    accuracy = [73.07, 99.31, 98.63]
    precision_attack = [98.0, 99.0, 99.0]
    recall_attack = [72.0, 100.0, 100.0]
    f1_attack = [83.0, 100.0, 99.0]
    
    # Create DataFrame
    performance_data = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision (Attack)': precision_attack,
        'Recall (Attack)': recall_attack,
        'F1-Score (Attack)': f1_attack
    })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.2
    
    # Create bars
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#2E8B57', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, precision_attack, width, label='Precision (Attack)', color='#4682B4', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, recall_attack, width, label='Recall (Attack)', color='#DC143C', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, f1_attack, width, label='F1-Score (Attack)', color='#FF8C00', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Performance Score (%)', fontweight='bold')
    ax.set_title('IoT Botnet Detection: Model Performance Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    
    plt.tight_layout()
    plt.savefig('results/model_performance_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("Performance comparison chart created")

def create_dataset_composition():
    """Create professional dataset composition visualization"""
    # Data composition
    benign_samples = 277965
    attack_samples = 3253333
    total_samples = benign_samples + attack_samples
    
    benign_percent = (benign_samples / total_samples) * 100
    attack_percent = (attack_samples / total_samples) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Donut chart
    sizes = [attack_percent, benign_percent]
    labels = ['Attack Traffic', 'Benign Traffic']
    colors = ['#FF6B6B', '#4ECDC4']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                                      colors=colors, explode=explode, pctdistance=0.85,
                                      wedgeprops=dict(width=0.5))
    
    # Draw center circle for donut effect
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax1.add_artist(centre_circle)
    
    ax1.set_title('Dataset Composition\n(N-BaLoT Dataset)', fontweight='bold', pad=20)
    
    # Format text
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Bar chart with sample counts
    categories = ['Benign Traffic', 'Attack Traffic']
    sample_counts = [benign_samples, attack_samples]
    colors_bar = ['#4ECDC4', '#FF6B6B']
    
    bars = ax2.bar(categories, sample_counts, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.8)
    
    ax2.set_title('Sample Distribution\n(Absolute Counts)', fontweight='bold', pad=20)
    ax2.set_ylabel('Number of Samples', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis to show values in millions
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50000,
                f'{height:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/dataset_composition_professional.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("Dataset composition visualization created")

def create_confusion_matrices():
    """Create confusion matrices for all three models"""
    # Data from experimental results
    models_data = {
        'Statistical-Only (Isolation Forest)': {
            'TN': 230547, 'FP': 47418, 'FN': 911731, 'TP': 2341602
        },
        'LSTM-Only (Deep Autoencoder)': {
            'TN': 252786, 'FP': 25179, 'FN': 24147, 'TP': 3229186
        },
        'Hybrid (Fusion Model)': {
            'TN': 230651, 'FP': 47314, 'FN': 31176, 'TP': 3222157
        }
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, data) in enumerate(models_data.items()):
        # Create confusion matrix
        cm = np.array([[data['TN'], data['FP']], 
                      [data['FN'], data['TP']]])
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        # Create heatmap
        sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=['Predicted Benign', 'Predicted Attack'],
                   yticklabels=['Actual Benign', 'Actual Attack'],
                   ax=axes[idx], cbar=True, square=True)
        
        axes[idx].set_title(f'{model_name}\nConfusion Matrix (%)', fontweight='bold', pad=15)
        axes[idx].set_xlabel('Predicted Label', fontweight='bold')
        axes[idx].set_ylabel('True Label', fontweight='bold')
        
        # Add count annotations
        for i in range(2):
            for j in range(2):
                axes[idx].text(j+0.5, i+0.75, f'({cm[i,j]:,})', 
                             ha='center', va='center', fontsize=9, color='darkred', 
                             fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("Confusion matrices created")

def create_training_efficiency():
    """Create training time and efficiency visualization"""
    models = ['Statistical-Only', 'LSTM-Only', 'Hybrid']
    training_times = [4.5, 15.0, 19.5]  # in seconds
    accuracies = [73.07, 99.31, 98.63]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training time comparison
    bars1 = ax1.bar(models, training_times, color=['#FF9999', '#66B2FF', '#99FF99'], 
                    alpha=0.8, edgecolor='black', linewidth=0.8)
    ax1.set_title('Training Time Comparison', fontweight='bold', pad=15)
    ax1.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Efficiency scatter plot (Accuracy vs Training Time)
    scatter = ax2.scatter(training_times, accuracies, s=200, 
                         c=['#FF9999', '#66B2FF', '#99FF99'], 
                         alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add model labels
    for i, model in enumerate(models):
        ax2.annotate(model, (training_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=10)
    
    ax2.set_title('Model Efficiency\n(Accuracy vs Training Time)', fontweight='bold', pad=15)
    ax2.set_xlabel('Training Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 25)
    ax2.set_ylim(70, 100)
    
    plt.tight_layout()
    plt.savefig('results/training_efficiency.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("Training efficiency visualization created")

def create_attack_type_breakdown():
    """Create attack type distribution visualization"""
    # Attack types present in N-BaLoT dataset
    gafgyt_types = ['combo', 'junk', 'scan', 'tcp', 'udp']
    mirai_types = ['ack', 'scan', 'syn', 'udp', 'udpplain']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gafgyt attacks pie chart
    gafgyt_counts = [len(gafgyt_types)] * len(gafgyt_types)  # Simplified for visualization
    colors1 = plt.cm.Reds(np.linspace(0.4, 0.8, len(gafgyt_types)))
    
    wedges1, texts1, autotexts1 = ax1.pie(gafgyt_counts, labels=gafgyt_types, autopct='%1.1f%%',
                                          colors=colors1, startangle=90)
    ax1.set_title('Gafgyt Botnet Attack Types', fontweight='bold', pad=20)
    
    # Mirai attacks pie chart  
    mirai_counts = [len(mirai_types)] * len(mirai_types)  # Simplified for visualization
    colors2 = plt.cm.Blues(np.linspace(0.4, 0.8, len(mirai_types)))
    
    wedges2, texts2, autotexts2 = ax2.pie(mirai_counts, labels=mirai_types, autopct='%1.1f%%',
                                          colors=colors2, startangle=90)
    ax2.set_title('Mirai Botnet Attack Types', fontweight='bold', pad=20)
    
    # Format text
    for texts in [texts1, texts2]:
        for text in texts:
            text.set_fontweight('bold')
            
    for autotexts in [autotexts1, autotexts2]:
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig('results/attack_type_breakdown.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("Attack type breakdown visualization created")

def main():
    """Generate all professional visualizations"""
    print("Generating Professional Research Visualizations...")
    print("-" * 55)
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Generate all plots
    create_performance_comparison()
    create_dataset_composition()
    create_confusion_matrices()
    create_training_efficiency()
    create_attack_type_breakdown()
    
    print("-" * 55)
    print("All professional visualizations created successfully!")
    print("Saved to: results/ directory")
    print("\nGenerated files:")
    print("  - model_performance_comparison.png")
    print("  - dataset_composition_professional.png")
    print("  - confusion_matrices_comparison.png")
    print("  - training_efficiency.png")
    print("  - attack_type_breakdown.png")

if __name__ == "__main__":
    main()