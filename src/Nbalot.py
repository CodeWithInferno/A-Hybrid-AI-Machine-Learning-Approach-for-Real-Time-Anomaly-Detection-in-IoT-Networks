"""
run_full_journal_experiment.py

This script conducts the definitive, full-scale experiment for the hybrid anomaly detection journal paper.
It performs the following steps:
1.  Configures professional logging to save all output to a timestamped file.
2.  Attempts to configure a GPU for TensorFlow.
3.  Loads a large, memory-efficient sample (50%) from the N-BaIoT dataset.
4.  Preprocesses the data.
5.  Trains and evaluates three models: Statistical-Only (Isolation Forest), LSTM-Only, and a fused Hybrid model.
6.  Generates and saves a suite of advanced and illustrative visualizations:
    - Dataset Composition Donut Chart
    - Comparative ROC Curve Plot
    - Hybrid Model Performance Scorecard
    - SHAP Summary Plot
    - SHAP Force Plots for individual predictions
    - t-SNE 2D Visualization of learned clusters
    - Comparative Performance Bar Chart
"""

import pandas as pd
import glob
import os
import numpy as np
import time
import tensorflow as tf
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix

# --- 1. SETUP LOGGING AND STYLES ---
log_filename = f"full_experiment_log_{{time.strftime('%Y-%m-%d_%H-%M-%S')}}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("Definitive Journal Experiment Started")
sns.set_theme(style="whitegrid") # Set a professional plot style

# --- 2. CONFIGURE GPU ---
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"--- GPU Detected and Configured: {len(gpus)} GPU(s) available ---")
        except RuntimeError as e:
            logging.error(e)
    else:
        logging.warning("--- No GPU detected. Running on CPU. This will be slow. ---")

# --- 3. DATA LOADING AND PREPROCESSING ---
def load_and_sample_data(path, sample_fraction=0.50): # STABLE SAMPLE FRACTION
    all_files = glob.glob(os.path.join(path, "*.csv"))
    logging.info(f"Found {len(all_files)} files. Sampling {sample_fraction*100}% from each...")
    
    df_list = []
    for i, filename in enumerate(all_files):
        if 'data_summary' in filename or 'device_info' in filename or 'features' in filename:
            continue
        
        logging.info(f"Processing file {i+1}/{len(all_files)}: {os.path.basename(filename)}")
        df_temp = pd.read_csv(filename)
        df_sample_temp = df_temp.sample(frac=sample_fraction, random_state=42)
        
        if 'benign' in filename:
            df_sample_temp['label'] = 0
        else:
            df_sample_temp['label'] = 1
            
        df_list.append(df_sample_temp)

    logging.info("Combining all samples...")
    df_sample = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined sample created successfully! Shape: {df_sample.shape}")
    return df_sample

def preprocess_data(df):
    logging.info("Preprocessing data...")
    X_sample = df.drop('label', axis=1)
    y_sample = df['label']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sample)
    logging.info(f"Data preprocessing complete. Final shape: {X_scaled.shape}")
    return X_scaled, y_sample, X_sample.columns

# --- 4. VISUALIZATION FUNCTIONS ---
def plot_dataset_composition(y_sample, filename="dataset_composition.png"):
    plt.figure(figsize=(8, 8))
    labels = ['Attack', 'Benign']
    sizes = y_sample.value_counts().values
    colors = ['#ff6666', '#66b3ff']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, colors=colors, textprops={'fontsize': 14})
    # Draw a circle at the center to make it a donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Dataset Composition', fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"Dataset composition plot saved to {filename}")
    plt.close()

def plot_performance_scorecard(report, filename="performance_scorecard.png"):
    metrics = ['precision', 'recall', 'f1-score']
    scores = [report['Attack (1)'][metric] for metric in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, scores, color=['#ff6666', '#ffcc66', '#66b3ff'])
    plt.ylabel('Score')
    plt.title('Hybrid Model Performance Scorecard (Attack Class)', fontsize=16)
    plt.ylim(0, 1.1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)
    plt.savefig(filename)
    logging.info(f"Performance scorecard plot saved to {filename}")
    plt.close()

def plot_tsne(X_scaled, y_sample, filename="tsne_visualization.png"):
    logging.info("Generating t-SNE visualization... (This is computationally intensive)")
    # t-SNE is very slow, so we use a smaller, balanced sub-sample
    n_samples_tsne = 5000
    benign_indices = np.where(y_sample == 0)[0]
    attack_indices = np.where(y_sample == 1)[0]
    sample_benign_indices = np.random.choice(benign_indices, n_samples_tsne // 2, replace=False)
    sample_attack_indices = np.random.choice(attack_indices, n_samples_tsne // 2, replace=False)
    tsne_indices = np.concatenate([sample_benign_indices, sample_attack_indices])
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, max_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(X_scaled[tsne_indices])
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=y_sample.iloc[tsne_indices].map({0: 'Benign', 1: 'Attack'}),
        palette=sns.color_palette(["#66b3ff", "#ff6666"]),
        legend="full",
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Data Clusters', fontsize=16)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(filename)
    logging.info(f"t-SNE plot saved to {filename}")
    plt.close()

def plot_comparison_chart(results, filename="model_comparison.png"):
    df = pd.DataFrame(results).set_index('Model')
    df_plot = df[['Accuracy', 'Precision (Attack)', 'Recall (Attack)', 'F1-Score (Attack)']]
    
    df_plot.plot(kind='bar', figsize=(14, 8), colormap='viridis')
    plt.title('Model Performance Comparison', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=0)
    plt.ylim(0.9, 1.0)
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.savefig(filename)
    logging.info(f"Model comparison plot saved to {filename}")
    plt.close()

# --- 5. MAIN EXPERIMENT FUNCTION ---
def main():
    start_time = time.time()
    
    configure_gpu()

    data_path = 'data/N-BaLot/'
    df_sample = load_and_sample_data(data_path, sample_fraction=0.50) # Using a 50% sample
    X_scaled, y_sample, feature_names = preprocess_data(df_sample)
    
    plot_dataset_composition(y_sample)

    X_normal = X_scaled[y_sample == 0]
    X_normal_reshaped = X_normal.reshape(X_normal.shape[0], 1, X_normal.shape[1])
    X_scaled_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    logging.info("--- Training Statistical-Only Model (Isolation Forest)... ---")
    iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(X_normal)
    statistical_scores = -iso_forest.decision_function(X_scaled)
    logging.info("Statistical model training complete.")

    logging.info("--- Training LSTM-Only Model... ---")
    input_dim = X_scaled.shape[1]
    latent_dim = 64
    inputs = Input(shape=(1, input_dim))
    encoder = LSTM(latent_dim, activation='relu')(inputs)
    decoder = RepeatVector(1)(encoder)
    decoder = LSTM(input_dim, activation='relu', return_sequences=True)(decoder)
    output = TimeDistributed(Dense(input_dim))(decoder)
    autoencoder = Model(inputs=inputs, outputs=output)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_normal_reshaped, X_normal_reshaped, epochs=10, batch_size=512, validation_split=0.1, verbose=1)
    logging.info("LSTM model training complete.")
    
    logging.info("Calculating LSTM scores using batch prediction to conserve memory...")
    predictions_lstm = autoencoder.predict(X_scaled_reshaped, batch_size=10000)
    lstm_scores = np.mean(np.power(X_scaled_reshaped - predictions_lstm, 2), axis=2).flatten()

    logging.info("--- Creating and Evaluating Hybrid Model... ---")
    scaler_scores = MinMaxScaler()
    scores_to_normalize = np.vstack((statistical_scores, lstm_scores)).T
    scaled_scores = scaler_scores.fit_transform(scores_to_normalize)
    composite_scores = (0.6 * scaled_scores[:, 1]) + (0.4 * scaled_scores[:, 0])

    models = { "Statistical-Only": statistical_scores, "LSTM-Only": lstm_scores, "Hybrid": composite_scores }
    
    all_results = []
    plt.figure(figsize=(10, 8))
    for model_name, scores in models.items():
        fpr, tpr, thresholds = roc_curve(y_sample, scores)
        auc_score = roc_auc_score(y_sample, scores)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        predictions = (scores > optimal_threshold).astype(int)
        
        accuracy = accuracy_score(y_sample, predictions)
        report = classification_report(y_sample, predictions, target_names=['Benign (0)', 'Attack (1)'], output_dict=True)
        
        all_results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision (Attack)': report['Attack (1)']['precision'],
            'Recall (Attack)': report['Attack (1)']['recall'],
            'F1-Score (Attack)': report['Attack (1)']['f1-score'],
            'AUC': auc_score
        })
        
        logging.info(f"\n{'='*20} RESULTS FOR: {model_name} {'='*20}")
        logging.info(f"Optimal Threshold: {optimal_threshold:.6f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info("\nClassification Report:\n" + classification_report(y_sample, predictions, target_names=['Benign (0)', 'Attack (1)']))
        
        if model_name == "Hybrid":
            plot_performance_scorecard(report)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparative ROC Curves for Anomaly Detection Models', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig("comparative_roc_curve.png")
    logging.info("Comparative ROC curve plot saved to comparative_roc_curve.png")
    plt.close()

    plot_comparison_chart(all_results)

    # --- 6. EXPLAINABLE AI (XAI) WITH SHAP ---
    logging.info("\n--- Generating SHAP Explanations for LSTM Model... ---")
    background_data = X_normal[np.random.choice(X_normal.shape[0], 100, replace=False)]
    
    benign_indices = np.where(y_sample == 0)[0]
    attack_indices = np.where(y_sample == 1)[0]
    sample_benign_indices = np.random.choice(benign_indices, 250, replace=False)
    sample_attack_indices = np.random.choice(attack_indices, 250, replace=False)
    explain_indices = np.concatenate([sample_benign_indices, sample_attack_indices])
    explain_data = X_scaled[explain_indices]
    
    def error_model(data_2d):
        data_3d = data_2d.reshape(data_2d.shape[0], 1, data_2d.shape[1])
        reconstructions = autoencoder.predict(data_3d, batch_size=10000)
        error = np.mean(np.power(data_3d - reconstructions, 2), axis=2).flatten()
        return error

    explainer = shap.KernelExplainer(error_model, background_data)
    shap_values = explainer.shap_values(explain_data)

    logging.info("SHAP values calculated. Generating summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, explain_data, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot for LSTM Reconstruction Error", fontsize=16)
    plt.savefig("shap_summary_plot.png", bbox_inches='tight')
    logging.info("SHAP summary plot saved to shap_summary_plot.png")
    plt.close()
    
    logging.info("Generating SHAP force plots for individual predictions...")
    # Explain one benign instance
    shap.force_plot(explainer.expected_value, shap_values[0,:], explain_data[0,:], feature_names=feature_names, matplotlib=True, show=False)
    plt.savefig("shap_force_plot_benign.png", bbox_inches='tight')
    plt.close()
    # Explain one attack instance (the first attack in our explain_data)
    shap.force_plot(explainer.expected_value, shap_values[250,:], explain_data[250,:], feature_names=feature_names, matplotlib=True, show=False)
    plt.savefig("shap_force_plot_attack.png", bbox_inches='tight')
    plt.close()
    logging.info("SHAP force plots saved.")

    # --- 7. t-SNE VISUALIZATION ---
    plot_tsne(X_scaled, y_sample)

    end_time = time.time()
    logging.info(f"Total experiment duration: {((end_time - start_time) / 60):.2f} minutes")

if __name__ == "__main__":
    try:
        import shap
    except ImportError:
        logging.info("Installing SHAP library...")
        os.system('pip install shap')
        import shap
    main()
