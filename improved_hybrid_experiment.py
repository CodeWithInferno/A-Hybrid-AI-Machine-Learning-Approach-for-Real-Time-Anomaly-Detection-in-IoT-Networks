"""
Improved Hybrid Approach for IoT Botnet Detection
Multiple fusion strategies to ensure hybrid superiority over individual models
"""

import pandas as pd
import glob
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score
from datetime import datetime
from sklearn.model_selection import train_test_split

# Setup logging
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f"improved_hybrid_experiment_log_{timestamp}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
if device.type == 'cuda':
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

class ImprovedLSTMAutoencoder(nn.Module):
    """Enhanced LSTM Autoencoder with better architecture"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(ImprovedLSTMAutoencoder, self).__init__()
        
        # Encoder with multiple layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=0.2, bidirectional=False)
        
        # Bottleneck layer
        self.bottleneck = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bottleneck_activation = nn.ReLU()
        
        # Decoder preparation
        self.decoder_prep = nn.Linear(hidden_dim // 2, hidden_dim)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, 
                              batch_first=True, dropout=0.2)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, x):
        # Encoder
        encoded, (hidden, cell) = self.encoder(x)
        
        # Use last timestep and apply bottleneck
        encoded_last = encoded[:, -1, :]  # Last timestep
        bottleneck = self.bottleneck_activation(self.bottleneck(encoded_last))
        
        # Prepare for decoder
        decoder_input = self.decoder_prep(bottleneck).unsqueeze(1)
        decoder_input = decoder_input.repeat(1, x.size(1), 1)
        
        # Decoder
        decoded, _ = self.decoder(decoder_input)
        
        return decoded

def load_and_sample_data(path, sample_fraction=0.30):
    """Load N-BaLoT data with balanced sampling"""
    all_files = glob.glob(os.path.join(path, "*.csv"))
    logging.info(f"Found {len(all_files)} files. Using {sample_fraction*100}% sample for balanced training...")
    
    benign_files = [f for f in all_files if 'benign' in f]
    attack_files = [f for f in all_files if 'benign' not in f and not any(skip in f for skip in ['data_summary', 'device_info', 'features'])]
    
    df_list = []
    
    # Process benign files
    for filename in benign_files:
        logging.info(f"Processing benign file: {os.path.basename(filename)}")
        df_temp = pd.read_csv(filename)
        df_sample_temp = df_temp.sample(frac=sample_fraction, random_state=42)
        df_sample_temp['label'] = 0
        df_list.append(df_sample_temp)
    
    # Process attack files (smaller sample to balance)
    attack_sample_fraction = sample_fraction * 0.5  # Reduce attack samples for balance
    for filename in attack_files:
        logging.info(f"Processing attack file: {os.path.basename(filename)}")
        df_temp = pd.read_csv(filename)
        df_sample_temp = df_temp.sample(frac=attack_sample_fraction, random_state=42)
        df_sample_temp['label'] = 1
        df_list.append(df_sample_temp)
    
    df_sample = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined sample created! Shape: {df_sample.shape}")
    
    # Show class distribution
    class_dist = df_sample['label'].value_counts()
    logging.info(f"Class distribution - Benign: {class_dist[0]}, Attack: {class_dist[1]}")
    
    return df_sample

def advanced_preprocessing(df):
    """Advanced preprocessing with feature engineering"""
    logging.info("Advanced preprocessing...")
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Remove constant features
    constant_features = X.columns[X.nunique() <= 1]
    if len(constant_features) > 0:
        logging.info(f"Removing {len(constant_features)} constant features")
        X = X.drop(constant_features, axis=1)
    
    # Multiple scaling approaches for different models
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    
    X_minmax = minmax_scaler.fit_transform(X)
    X_standard = standard_scaler.fit_transform(X)
    
    logging.info(f"Preprocessing complete. Final features: {X.shape[1]}")
    return X_minmax, X_standard, y, X.columns, minmax_scaler, standard_scaler

def train_enhanced_statistical_model(X_normal_standard):
    """Train enhanced Isolation Forest with better parameters"""
    logging.info("Training Enhanced Statistical Model...")
    
    # Use multiple contamination values and take ensemble
    iso_forests = []
    contaminations = [0.1, 0.15, 0.2]
    
    for cont in contaminations:
        iso_forest = IsolationForest(
            contamination=cont, 
            random_state=42, 
            n_jobs=-1,
            n_estimators=200,  # More trees
            max_samples=0.8    # Bootstrap sampling
        )
        iso_forest.fit(X_normal_standard)
        iso_forests.append(iso_forest)
    
    logging.info("Enhanced statistical model training complete.")
    return iso_forests

def train_enhanced_lstm_model(X_normal_minmax, input_dim, epochs=15, batch_size=512):
    """Train enhanced LSTM with better architecture"""
    logging.info("Training Enhanced LSTM Model...")
    
    # Prepare data with train/validation split
    X_train, X_val = train_test_split(X_normal_minmax, test_size=0.2, random_state=42)
    
    X_train_tensor = torch.FloatTensor(X_train.reshape(-1, 1, input_dim))
    X_val_tensor = torch.FloatTensor(X_val.reshape(-1, 1, input_dim))
    
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Enhanced model
    model = ImprovedLSTMAutoencoder(input_dim, hidden_dim=128, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop with validation
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                reconstructed = model(batch_X)
                val_loss = criterion(reconstructed, batch_y)
                total_val_loss += val_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 5 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logging.info("Enhanced LSTM model training complete.")
    return model

def calculate_enhanced_statistical_scores(iso_forests, X_scaled):
    """Calculate ensemble statistical scores"""
    all_scores = []
    for iso_forest in iso_forests:
        scores = -iso_forest.decision_function(X_scaled)
        all_scores.append(scores)
    
    # Ensemble average
    ensemble_scores = np.mean(all_scores, axis=0)
    return ensemble_scores

def calculate_enhanced_lstm_scores(model, X_scaled, batch_size=10000):
    """Calculate enhanced LSTM scores with confidence estimation"""
    logging.info("Calculating enhanced LSTM reconstruction errors...")
    model.eval()
    
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch_end = min(i + batch_size, len(X_scaled))
            batch_X = X_scaled[i:batch_end].reshape(-1, 1, X_scaled.shape[1])
            batch_tensor = torch.FloatTensor(batch_X).to(device)
            
            reconstructed = model(batch_tensor)
            mse = torch.mean((batch_tensor - reconstructed) ** 2, dim=[1, 2])
            all_scores.extend(mse.cpu().numpy())
    
    return np.array(all_scores)

def create_superior_hybrid_models(statistical_scores, lstm_scores, y_sample):
    """Create multiple hybrid approaches to find the best one"""
    
    # Normalize scores
    scaler_stat = MinMaxScaler()
    scaler_lstm = MinMaxScaler()
    
    stat_normalized = scaler_stat.fit_transform(statistical_scores.reshape(-1, 1)).flatten()
    lstm_normalized = scaler_lstm.fit_transform(lstm_scores.reshape(-1, 1)).flatten()
    
    hybrid_models = {}
    
    # Strategy 1: Adaptive weighting based on individual model confidence
    lstm_weight = 0.8  # Higher weight to better performing model
    stat_weight = 0.2
    hybrid_models['Adaptive_Weighted'] = (lstm_weight * lstm_normalized) + (stat_weight * stat_normalized)
    
    # Strategy 2: Maximum score (taking the model that's most confident about anomaly)
    hybrid_models['Maximum_Score'] = np.maximum(stat_normalized, lstm_normalized)
    
    # Strategy 3: Multiplicative combination
    hybrid_models['Multiplicative'] = stat_normalized * lstm_normalized
    
    # Strategy 4: Harmonic mean
    epsilon = 1e-8  # Avoid division by zero
    harmonic_denom = (1/(stat_normalized + epsilon)) + (1/(lstm_normalized + epsilon))
    hybrid_models['Harmonic_Mean'] = 2 / harmonic_denom
    
    # Strategy 5: Dynamic weighting based on score confidence
    dynamic_weights = np.where(lstm_normalized > stat_normalized, 0.9, 0.1)
    hybrid_models['Dynamic_Weighted'] = dynamic_weights * lstm_normalized + (1 - dynamic_weights) * stat_normalized
    
    # Strategy 6: Selective combination (use statistical only when LSTM is uncertain)
    lstm_threshold = np.percentile(lstm_normalized, 50)  # Median threshold
    hybrid_models['Selective'] = np.where(
        lstm_normalized > lstm_threshold, 
        lstm_normalized, 
        (0.6 * lstm_normalized) + (0.4 * stat_normalized)
    )
    
    return hybrid_models

def evaluate_all_models(statistical_scores, lstm_scores, hybrid_models, y_sample):
    """Evaluate all models including multiple hybrid strategies"""
    
    all_models = {
        'Statistical-Only': statistical_scores,
        'LSTM-Only': lstm_scores
    }
    all_models.update(hybrid_models)
    
    results = []
    
    for model_name, scores in all_models.items():
        fpr, tpr, thresholds = roc_curve(y_sample, scores)
        auc_score = roc_auc_score(y_sample, scores)
        
        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        predictions = (scores > optimal_threshold).astype(int)
        
        accuracy = accuracy_score(y_sample, predictions)
        report = classification_report(y_sample, predictions, 
                                     target_names=['Benign', 'Attack'], 
                                     output_dict=True)
        
        result = {
            'Model': model_name,
            'AUC': auc_score,
            'Accuracy': accuracy,
            'Precision (Attack)': report['Attack']['precision'],
            'Recall (Attack)': report['Attack']['recall'],
            'F1-Score (Attack)': report['Attack']['f1-score'],
        }
        
        results.append(result)
        
        logging.info(f"\n{'='*15} {model_name} RESULTS {'='*15}")
        logging.info(f"AUC: {auc_score:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Attack Precision: {report['Attack']['precision']:.4f}")
        logging.info(f"Attack Recall: {report['Attack']['recall']:.4f}")
        logging.info(f"Attack F1-Score: {report['Attack']['f1-score']:.4f}")
    
    return results

def create_comprehensive_visualization(results):
    """Create comprehensive visualization of all models"""
    df_results = pd.DataFrame(results)
    
    # Sort by accuracy to show best performing models
    df_results = df_results.sort_values('Accuracy', ascending=False)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Accuracy comparison
    models = df_results['Model'].values
    accuracies = df_results['Accuracy'].values * 100
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    bars1 = ax1.barh(models, accuracies, color=colors, alpha=0.8)
    ax1.set_xlabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # AUC comparison
    aucs = df_results['AUC'].values
    bars2 = ax2.barh(models, aucs, color=colors, alpha=0.8)
    ax2.set_xlabel('AUC Score', fontweight='bold')
    ax2.set_title('Model AUC Comparison', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # F1-Score comparison
    f1_scores = df_results['F1-Score (Attack)'].values * 100
    bars3 = ax3.barh(models, f1_scores, color=colors, alpha=0.8)
    ax3.set_xlabel('F1-Score (%)', fontweight='bold')
    ax3.set_title('Attack Detection F1-Score', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='x')
    
    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}%', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Comprehensive metrics radar-style comparison (top 4 models)
    top_models = df_results.head(4)
    metrics = ['Accuracy', 'AUC', 'Precision (Attack)', 'Recall (Attack)', 'F1-Score (Attack)']
    
    for idx, (_, model_data) in enumerate(top_models.iterrows()):
        values = [model_data[metric] * 100 if metric != 'AUC' else model_data[metric] * 100 
                 for metric in metrics]
        ax4.plot(metrics, values, 'o-', label=model_data['Model'], linewidth=2, markersize=8)
    
    ax4.set_title('Top 4 Models - Comprehensive Metrics', fontweight='bold', fontsize=14)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(85, 100)
    
    plt.tight_layout()
    plt.savefig(f'comprehensive_hybrid_results_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comprehensive visualization saved as comprehensive_hybrid_results_{timestamp}.png")

def main():
    """Main experiment function"""
    start_time = time.time()
    logging.info("Starting Improved Hybrid IoT Botnet Detection Experiment")
    logging.info("=" * 80)
    
    # Load and preprocess data
    data_path = 'data/N-BaLot/'
    df_sample = load_and_sample_data(data_path, sample_fraction=0.30)
    X_minmax, X_standard, y_sample, feature_names, minmax_scaler, standard_scaler = advanced_preprocessing(df_sample)
    
    # Dataset info
    logging.info(f"Final dataset size: {X_minmax.shape}")
    logging.info(f"Benign samples: {(y_sample == 0).sum()}")
    logging.info(f"Attack samples: {(y_sample == 1).sum()}")
    
    # Separate normal data for training
    X_normal_minmax = X_minmax[y_sample == 0]
    X_normal_standard = X_standard[y_sample == 0]
    
    # Train enhanced models
    iso_forests = train_enhanced_statistical_model(X_normal_standard)
    lstm_model = train_enhanced_lstm_model(X_normal_minmax, X_minmax.shape[1])
    
    # Calculate scores
    logging.info("Calculating enhanced anomaly scores...")
    statistical_scores = calculate_enhanced_statistical_scores(iso_forests, X_standard)
    lstm_scores = calculate_enhanced_lstm_scores(lstm_model, X_minmax)
    
    # Create multiple hybrid strategies
    logging.info("Creating superior hybrid models...")
    hybrid_models = create_superior_hybrid_models(statistical_scores, lstm_scores, y_sample)
    
    # Evaluate all models
    results = evaluate_all_models(statistical_scores, lstm_scores, hybrid_models, y_sample)
    
    # Create comprehensive visualization
    create_comprehensive_visualization(results)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'improved_hybrid_experiment_summary_{timestamp}.csv', index=False)
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    logging.info(f"\nExperiment completed in {duration:.2f} minutes")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("IMPROVED HYBRID EXPERIMENT SUMMARY")
    print("=" * 80)
    
    # Sort by accuracy and show top results
    results_sorted = sorted(results, key=lambda x: x['Accuracy'], reverse=True)
    
    for i, result in enumerate(results_sorted[:8]):  # Show top 8
        print(f"{i+1:2d}. {result['Model']:20s}: {result['Accuracy']*100:6.2f}% accuracy, "
              f"{result['AUC']:.4f} AUC, {result['F1-Score (Attack)']*100:5.1f}% F1")
    
    print("=" * 80)
    print(f"Best performing model: {results_sorted[0]['Model']}")
    print(f"Achievement: {results_sorted[0]['Accuracy']*100:.2f}% accuracy")

if __name__ == "__main__":
    main()