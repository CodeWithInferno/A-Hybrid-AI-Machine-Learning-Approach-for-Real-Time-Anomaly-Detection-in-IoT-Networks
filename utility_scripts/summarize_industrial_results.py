"""
Summarize Industrial IoT Dataset Results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Summary of datasets created
datasets_info = {
    'NASA C-MAPSS': {
        'samples': 17604,
        'sensors': 21,
        'type': 'Turbofan Engine',
        'features': ['Temperature', 'Pressure', 'Speed', 'Flow'],
        'use_case': 'Predictive Maintenance'
    },
    'CWRU Bearing': {
        'samples': 480000,  # 4 conditions × 120k samples each
        'sensors': 1,
        'type': 'Vibration',
        'features': ['Acceleration'],
        'use_case': 'Fault Diagnosis'
    },
    'Multi-Sensor Industrial': {
        'samples': 100000,
        'sensors': 11,
        'type': 'Multi-Modal',
        'features': ['Temperature', 'Pressure', 'Vibration', 'Flow', 'Current', 'RPM'],
        'use_case': 'Anomaly Detection'
    }
}

# Create comparison with N-BaLoT
comparison_data = {
    'Dataset': ['N-BaLoT (Network)', 'Industrial Multi-Sensor'],
    'Data Type': ['Network Traffic', 'Physical Sensors'],
    'Sensors': ['115 features', '11 sensors'],
    'Samples': ['1.14M', '100K'],
    'Modalities': ['Single (Network)', 'Multi-Modal'],
    'Real-time': ['Packet Analysis', 'Sensor Streaming'],
    'Application': ['Botnet Detection', 'Predictive Maintenance']
}

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Dataset Sizes Comparison
datasets = list(datasets_info.keys())
samples = [datasets_info[d]['samples'] for d in datasets]
sensors = [datasets_info[d]['sensors'] for d in datasets]

x = np.arange(len(datasets))
width = 0.35

ax1.bar(x - width/2, samples, width, label='Samples', color='skyblue')
ax1.set_ylabel('Number of Samples')
ax1.set_xlabel('Dataset')
ax1.set_title('Industrial IoT Datasets - Sample Sizes', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets)
ax1.legend()

# Add sensor count on secondary y-axis
ax1_twin = ax1.twinx()
ax1_twin.bar(x + width/2, sensors, width, label='Sensors', color='lightcoral')
ax1_twin.set_ylabel('Number of Sensors')
ax1_twin.legend(loc='upper right')

# 2. Sensor Types Distribution
all_features = []
for d in datasets_info.values():
    all_features.extend(d['features'])

feature_counts = {}
for f in all_features:
    feature_counts[f] = feature_counts.get(f, 0) + 1

ax2.pie(feature_counts.values(), labels=feature_counts.keys(), autopct='%1.1f%%', startangle=90)
ax2.set_title('Distribution of Sensor Types Across Datasets', fontweight='bold')

# 3. Comparison Table
table_data = []
for key in comparison_data:
    if key != 'Dataset':
        row = [comparison_data[key][0], comparison_data[key][1]]
        table_data.append(row)

ax3.axis('tight')
ax3.axis('off')
table = ax3.table(cellText=table_data,
                  rowLabels=list(comparison_data.keys())[1:],
                  colLabels=['N-BaLoT (IoT Network)', 'Industrial Sensors'],
                  cellLoc='left',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax3.set_title('IoT Security: Network vs Industrial Sensors', fontweight='bold', pad=20)

# 4. Expected Performance Improvements
methods = ['Network Only\n(N-BaLoT)', 'Sensor Only\n(Industrial)', 'Hybrid Fusion\n(Combined)']
accuracy = [99.47, 92.5, 99.8]  # Projected values
f1_scores = [99.47, 85.0, 99.5]  # Projected values

x = np.arange(len(methods))
width = 0.35

ax4.bar(x - width/2, accuracy, width, label='Accuracy', color='#2ecc71')
ax4.bar(x + width/2, f1_scores, width, label='F1-Score', color='#3498db')
ax4.set_ylabel('Performance (%)')
ax4.set_title('Expected Performance: Hybrid Approach Benefits', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(methods)
ax4.legend()
ax4.set_ylim(80, 100)
ax4.grid(True, alpha=0.3)

# Add value labels
for i, (acc, f1) in enumerate(zip(accuracy, f1_scores)):
    ax4.text(i - width/2, acc + 0.5, f'{acc:.1f}%', ha='center', va='bottom')
    ax4.text(i + width/2, f1 + 0.5, f'{f1:.1f}%', ha='center', va='bottom')

plt.suptitle('Industrial IoT Sensor Datasets for Hybrid Anomaly Detection', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('industrial_datasets_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary report
print("="*60)
print("INDUSTRIAL SENSOR DATASETS CREATED")
print("="*60)
print()
print("1. NASA C-MAPSS Turbofan Dataset:")
print(f"   - {datasets_info['NASA C-MAPSS']['samples']:,} samples")
print(f"   - {datasets_info['NASA C-MAPSS']['sensors']} sensor measurements")
print("   - Run-to-failure degradation data")
print("   - Perfect for predictive maintenance")
print()
print("2. CWRU Bearing Vibration Dataset:")
print(f"   - {datasets_info['CWRU Bearing']['samples']:,} vibration samples")
print("   - Multiple fault conditions")
print("   - 12kHz sampling rate")
print("   - Industry standard for bearing diagnostics")
print()
print("3. Multi-Sensor Industrial Dataset:")
print(f"   - {datasets_info['Multi-Sensor Industrial']['samples']:,} samples")
print(f"   - {datasets_info['Multi-Sensor Industrial']['sensors']} different sensors")
print("   - Temperature, Pressure, Vibration, Flow, Current, RPM")
print("   - Built-in anomaly scenarios")
print()
print("ADVANTAGES OVER NETWORK-ONLY APPROACH:")
print("- Physical sensor data captures real equipment behavior")
print("- Multi-modal fusion enables better anomaly detection")
print("- Time-series patterns reveal degradation trends")
print("- Direct correlation to equipment failures")
print()
print("HYBRID APPROACH BENEFITS:")
print("- Combine network behavior with physical symptoms")
print("- Earlier detection of cyber-physical attacks")
print("- Reduced false positives through sensor validation")
print("- Comprehensive industrial IoT security")

# Save summary to file
summary_path = Path("industrial_sensor_data/dataset_summary.txt")
summary_path.parent.mkdir(exist_ok=True)

with open(summary_path, 'w') as f:
    f.write("Industrial IoT Sensor Datasets Summary\n")
    f.write("="*50 + "\n\n")
    
    for name, info in datasets_info.items():
        f.write(f"{name}:\n")
        f.write(f"  Samples: {info['samples']:,}\n")
        f.write(f"  Sensors: {info['sensors']}\n")
        f.write(f"  Type: {info['type']}\n")
        f.write(f"  Features: {', '.join(info['features'])}\n")
        f.write(f"  Use Case: {info['use_case']}\n\n")

print(f"\nSummary saved to: {summary_path}")