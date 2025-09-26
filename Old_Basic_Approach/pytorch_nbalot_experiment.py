"""
PyTorch-based N-BaLoT Experiment for IoT Botnet Detection
Uses PyTorch instead of TensorFlow for GPU acceleration
Implements the same hybrid approach but with PyTorch backend
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score
from datetime import datetime

# Setup logging
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f"pytorch_experiment_log_{timestamp}.txt"
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
    logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class LSTMAutoencoder(nn.Module):
    """PyTorch LSTM Autoencoder for anomaly detection"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, 
                              batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, x):
        # Encoder
        encoded, (hidden, cell) = self.encoder(x)
        
        # Use the last hidden state
        encoded = encoded[:, -1, :].unsqueeze(1)  # Take last timestep
        
        # Repeat for sequence length
        encoded = encoded.repeat(1, x.size(1), 1)
        
        # Decoder
        decoded, _ = self.decoder(encoded)
        
        return decoded

def load_and_sample_data(path, sample_fraction=0.50):
    """Load and sample N-BaLoT data"""
    all_files = glob.glob(os.path.join(path, "*.csv"))
    logging.info(f"Found {len(all_files)} files. Sampling {sample_fraction*100}% from each...")
    
    df_list = []
    for i, filename in enumerate(all_files):
        if any(skip in filename for skip in ['data_summary', 'device_info', 'features']):
            continue
            
        logging.info(f"Processing file {i+1}/{len(all_files)}: {os.path.basename(filename)}")
        df_temp = pd.read_csv(filename)
        df_sample_temp = df_temp.sample(frac=sample_fraction, random_state=42)
        
        # Label creation
        df_sample_temp['label'] = 0 if 'benign' in filename else 1
        df_list.append(df_sample_temp)
    
    df_sample = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined sample created! Shape: {df_sample.shape}")
    return df_sample

def preprocess_data(df):
    """Preprocess the data"""
    logging.info("Preprocessing data...")
    X = df.drop('label', axis=1)
    y = df['label']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    logging.info(f"Data preprocessing complete. Shape: {X_scaled.shape}")
    return X_scaled, y, X.columns, scaler

def train_statistical_model(X_normal):
    """Train Isolation Forest model"""
    logging.info("Training Statistical-Only Model (Isolation Forest)...")
    iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(X_normal)
    logging.info("Statistical model training complete.")
    return iso_forest

def train_lstm_model(X_normal, input_dim, epochs=10, batch_size=512):
    """Train PyTorch LSTM Autoencoder"""
    logging.info("Training LSTM-Only Model (PyTorch)...")
    
    # Prepare data
    X_normal_tensor = torch.FloatTensor(X_normal.reshape(-1, 1, input_dim))
    dataset = TensorDataset(X_normal_tensor, X_normal_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = LSTMAutoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    logging.info("LSTM model training complete.")
    return model

def calculate_lstm_scores(model, X_scaled, batch_size=10000):
    """Calculate LSTM anomaly scores"""
    logging.info("Calculating LSTM reconstruction errors...")
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

def evaluate_models(statistical_scores, lstm_scores, y_sample):
    """Evaluate all three models and return results"""
    # Normalize scores for hybrid model
    scaler_scores = MinMaxScaler()
    scores_combined = np.vstack((statistical_scores, lstm_scores)).T
    scaled_scores = scaler_scores.fit_transform(scores_combined)
    
    # Create hybrid scores (60% LSTM, 40% Statistical)
    hybrid_scores = (0.6 * scaled_scores[:, 1]) + (0.4 * scaled_scores[:, 0])
    
    models = {
        "Statistical-Only": statistical_scores,
        "LSTM-Only": lstm_scores, 
        "Hybrid": hybrid_scores
    }
    
    results = []
    
    # Evaluate each model
    for model_name, scores in models.items():
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
            'Optimal_Threshold': optimal_threshold
        }
        
        results.append(result)
        
        logging.info(f"\n{'='*20} {model_name} RESULTS {'='*20}")
        logging.info(f"AUC: {auc_score:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Optimal Threshold: {optimal_threshold:.6f}")
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_sample, predictions, 
                                         target_names=['Benign', 'Attack']))
    
    return results

def create_results_visualizations(results, y_sample):
    """Create professional visualizations"""
    
    # Performance comparison
    df_results = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison
    models = df_results['Model'].values
    accuracies = df_results['Accuracy'].values * 100
    
    bars = ax1.bar(models, accuracies, color=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.8)
    ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_ylim(70, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # AUC comparison
    aucs = df_results['AUC'].values
    bars2 = ax2.bar(models, aucs, color=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.8)
    ax2.set_title('Model AUC Comparison', fontweight='bold', fontsize=14)
    ax2.set_ylabel('AUC Score', fontweight='bold')
    ax2.set_ylim(0.95, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'pytorch_experiment_results_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Results visualization saved as pytorch_experiment_results_{timestamp}.png")

def main():
    """Main experiment function"""
    start_time = time.time()
    logging.info("Starting PyTorch-based N-BaLoT Experiment")
    logging.info("=" * 60)
    
    # Load and preprocess data
    data_path = 'data/N-BaLot/'
    df_sample = load_and_sample_data(data_path, sample_fraction=0.50)
    X_scaled, y_sample, feature_names, scaler = preprocess_data(df_sample)
    
    # Dataset info
    logging.info(f"Dataset size: {X_scaled.shape}")
    logging.info(f"Benign samples: {(y_sample == 0).sum()}")
    logging.info(f"Attack samples: {(y_sample == 1).sum()}")
    
    # Separate normal data for training
    X_normal = X_scaled[y_sample == 0]
    logging.info(f"Normal data for training: {X_normal.shape}")
    
    # Train models
    iso_forest = train_statistical_model(X_normal)
    lstm_model = train_lstm_model(X_normal, X_scaled.shape[1])
    
    # Calculate scores
    logging.info("Calculating anomaly scores...")
    statistical_scores = -iso_forest.decision_function(X_scaled)
    lstm_scores = calculate_lstm_scores(lstm_model, X_scaled)
    
    # Evaluate models
    results = evaluate_models(statistical_scores, lstm_scores, y_sample)
    
    # Create visualizations
    create_results_visualizations(results, y_sample)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'pytorch_experiment_summary_{timestamp}.csv', index=False)
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    logging.info(f"\nExperiment completed in {duration:.2f} minutes")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for _, result in results_df.iterrows():
        print(f"{result['Model']:15s}: {result['Accuracy']*100:6.2f}% accuracy, {result['AUC']:.4f} AUC")
    print("=" * 60)

if __name__ == "__main__":
    main()