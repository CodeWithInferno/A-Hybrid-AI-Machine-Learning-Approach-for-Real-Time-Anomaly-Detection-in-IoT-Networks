"""
Multi-Scale Temporal Analyzer for IoT Anomaly Detection
Analyzes patterns from microseconds to hours simultaneously
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
import pywt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class MultiScaleTemporalAnalyzer:
    """
    Analyzes IoT data across multiple temporal scales to detect anomalies
    that manifest differently at various time resolutions.
    """
    
    def __init__(self, scales: List[str] = None):
        """
        Initialize multi-scale analyzer
        
        Args:
            scales: List of temporal scales to analyze
                   Default: ['microsecond', 'millisecond', 'second', 'minute', 'hour']
        """
        self.scales = scales or ['microsecond', 'millisecond', 'second', 'minute', 'hour']
        self.scale_factors = {
            'microsecond': 1e-6,
            'millisecond': 1e-3,
            'second': 1,
            'minute': 60,
            'hour': 3600
        }
        
        # Wavelet configurations for each scale
        self.wavelets = {
            'microsecond': 'db4',  # Daubechies for fine details
            'millisecond': 'sym5',  # Symlets for medium details
            'second': 'coif3',      # Coiflets for smooth patterns
            'minute': 'dmey',       # Discrete Meyer for trends
            'hour': 'mexh'          # Mexican hat for large patterns
        }
        
        self.scale_features = {}
        self.anomaly_scores = {}
        
    def decompose_signal(self, signal_data: np.ndarray, 
                        sampling_rate: float) -> Dict[str, np.ndarray]:
        """
        Decompose signal into multiple temporal scales using wavelets
        """
        decompositions = {}
        
        for scale in self.scales:
            # Select appropriate wavelet
            wavelet = self.wavelets[scale]
            
            # Determine decomposition level based on scale
            max_level = pywt.dwt_max_level(len(signal_data), wavelet)
            scale_level = self._get_scale_level(scale, sampling_rate, max_level)
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal_data, wavelet, level=scale_level)
            
            # Reconstruct at specific scale
            reconstructed = self._reconstruct_at_scale(coeffs, wavelet, scale_level)
            decompositions[scale] = reconstructed
            
        return decompositions
    
    def _get_scale_level(self, scale: str, sampling_rate: float, 
                        max_level: int) -> int:
        """Calculate appropriate wavelet level for given temporal scale"""
        scale_seconds = self.scale_factors[scale]
        scale_samples = scale_seconds * sampling_rate
        
        # Map to wavelet levels (each level ~doubles the scale)
        level = int(np.log2(scale_samples))
        return min(max(1, level), max_level)
    
    def _reconstruct_at_scale(self, coeffs: List, wavelet: str, 
                             target_level: int) -> np.ndarray:
        """Reconstruct signal at specific scale by zeroing other coefficients"""
        # Create a copy and zero out other scales
        scale_coeffs = [np.zeros_like(c) if i != target_level else c 
                       for i, c in enumerate(coeffs)]
        
        # Reconstruct
        return pywt.waverec(scale_coeffs, wavelet, mode='symmetric')
    
    def extract_scale_features(self, decompositions: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Extract temporal features at each scale
        """
        features = {}
        
        for scale, signal in decompositions.items():
            scale_features = {
                # Statistical features
                'mean': np.mean(signal),
                'std': np.std(signal),
                'skewness': self._compute_skewness(signal),
                'kurtosis': self._compute_kurtosis(signal),
                
                # Energy features
                'energy': np.sum(signal ** 2),
                'entropy': entropy(np.histogram(signal, bins=50)[0]),
                
                # Temporal features
                'zero_crossings': self._count_zero_crossings(signal),
                'peak_count': self._count_peaks(signal),
                'trend_strength': self._compute_trend_strength(signal),
                
                # Frequency features for this scale
                'dominant_freq': self._get_dominant_frequency(signal),
                'spectral_entropy': self._compute_spectral_entropy(signal),
                
                # Complexity measures
                'sample_entropy': self._sample_entropy(signal),
                'hurst_exponent': self._hurst_exponent(signal)
            }
            
            features[scale] = scale_features
            
        return features
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of distribution"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _count_zero_crossings(self, signal: np.ndarray) -> int:
        """Count number of zero crossings in signal"""
        return len(np.where(np.diff(np.sign(signal)))[0])
    
    def _count_peaks(self, signal: np.ndarray) -> int:
        """Count number of peaks in signal"""
        peaks, _ = signal.find_peaks(signal)
        return len(peaks)
    
    def _compute_trend_strength(self, signal: np.ndarray) -> float:
        """Compute strength of linear trend"""
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, 1)
        trend = coeffs[0] * x + coeffs[1]
        return np.corrcoef(signal, trend)[0, 1] ** 2
    
    def _get_dominant_frequency(self, signal: np.ndarray) -> float:
        """Get dominant frequency using FFT"""
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        dominant_idx = np.argmax(np.abs(fft[:len(fft)//2]))
        return abs(freqs[dominant_idx])
    
    def _compute_spectral_entropy(self, signal: np.ndarray) -> float:
        """Compute spectral entropy"""
        fft = np.fft.fft(signal)
        psd = np.abs(fft[:len(fft)//2]) ** 2
        psd_norm = psd / np.sum(psd)
        return entropy(psd_norm)
    
    def _sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy"""
        N = len(signal)
        r = r * np.std(signal)
        
        def _maxdist(xi, xj, m):
            return max([abs(float(xi[k]) - float(xj[k])) for k in range(m)])
        
        def _phi(m):
            patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
            C = 0
            for i in range(len(patterns)):
                matching = sum([1 for j in range(len(patterns)) 
                              if i != j and _maxdist(patterns[i], patterns[j], m) <= r])
                if matching > 0:
                    C += np.log(matching / (N - m))
            return C / (N - m + 1)
        
        try:
            return -np.log(_phi(m+1) / _phi(m))
        except:
            return 0
    
    def _hurst_exponent(self, signal: np.ndarray) -> float:
        """Estimate Hurst exponent using R/S analysis"""
        N = len(signal)
        if N < 100:
            return 0.5  # Default to random walk
        
        max_k = int(np.log2(N)) - 4
        R_S = []
        n_list = []
        
        for k in range(4, max_k):
            n = 2**k
            n_list.append(n)
            
            # Calculate R/S for this n
            rs_values = []
            for start in range(0, N-n, n//2):
                subseries = signal[start:start+n]
                mean = np.mean(subseries)
                y = np.cumsum(subseries - mean)
                R = np.max(y) - np.min(y)
                S = np.std(subseries, ddof=1)
                
                if S > 0:
                    rs_values.append(R/S)
            
            if rs_values:
                R_S.append(np.mean(rs_values))
        
        if len(R_S) > 1:
            log_n = np.log(n_list)
            log_RS = np.log(R_S)
            hurst = np.polyfit(log_n, log_RS, 1)[0]
            return np.clip(hurst, 0, 1)
        
        return 0.5
    
    def detect_scale_anomalies(self, features: Dict[str, Dict], 
                              historical_stats: Optional[Dict] = None) -> Dict[str, float]:
        """
        Detect anomalies at each temporal scale
        """
        anomaly_scores = {}
        
        for scale, scale_features in features.items():
            if historical_stats and scale in historical_stats:
                # Compare with historical baseline
                stats = historical_stats[scale]
                
                # Z-score based anomaly detection
                deviations = []
                for feature, value in scale_features.items():
                    if feature in stats:
                        mean = stats[feature]['mean']
                        std = stats[feature]['std']
                        if std > 0:
                            z_score = abs((value - mean) / std)
                            deviations.append(z_score)
                
                # Combine deviations
                anomaly_scores[scale] = np.mean(deviations) if deviations else 0
            else:
                # Use complexity-based anomaly score
                complexity = (scale_features['sample_entropy'] + 
                            abs(scale_features['hurst_exponent'] - 0.5) +
                            scale_features['spectral_entropy'])
                anomaly_scores[scale] = complexity / 3
        
        return anomaly_scores
    
    def fuse_multi_scale_scores(self, scale_scores: Dict[str, float], 
                               attack_type: Optional[str] = None) -> float:
        """
        Fuse anomaly scores from multiple scales based on attack characteristics
        """
        if attack_type:
            # Attack-specific weighting
            weights = self._get_attack_weights(attack_type)
        else:
            # Adaptive weighting based on score distribution
            weights = self._adaptive_weights(scale_scores)
        
        # Weighted fusion
        total_score = 0
        total_weight = 0
        
        for scale, score in scale_scores.items():
            if scale in weights:
                total_score += score * weights[scale]
                total_weight += weights[scale]
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _get_attack_weights(self, attack_type: str) -> Dict[str, float]:
        """Get scale weights based on attack characteristics"""
        attack_weights = {
            'ddos': {  # DDoS shows at second/minute scales
                'microsecond': 0.05,
                'millisecond': 0.1,
                'second': 0.4,
                'minute': 0.35,
                'hour': 0.1
            },
            'scanning': {  # Port scanning shows at millisecond/second
                'microsecond': 0.1,
                'millisecond': 0.4,
                'second': 0.35,
                'minute': 0.1,
                'hour': 0.05
            },
            'exfiltration': {  # Data theft shows at minute/hour scales
                'microsecond': 0.05,
                'millisecond': 0.05,
                'second': 0.2,
                'minute': 0.4,
                'hour': 0.3
            },
            'botnet': {  # Botnet C&C shows across scales
                'microsecond': 0.1,
                'millisecond': 0.2,
                'second': 0.3,
                'minute': 0.25,
                'hour': 0.15
            }
        }
        
        return attack_weights.get(attack_type, {scale: 0.2 for scale in self.scales})
    
    def _adaptive_weights(self, scale_scores: Dict[str, float]) -> Dict[str, float]:
        """Compute adaptive weights based on score distribution"""
        scores = np.array(list(scale_scores.values()))
        
        # Scales with higher relative scores get more weight
        if np.std(scores) > 0:
            normalized = (scores - np.mean(scores)) / np.std(scores)
            weights = np.exp(normalized) / np.sum(np.exp(normalized))  # Softmax
        else:
            weights = np.ones(len(scores)) / len(scores)
        
        return {scale: weight for scale, weight in zip(scale_scores.keys(), weights)}
    
    def visualize_multi_scale_analysis(self, decompositions: Dict[str, np.ndarray],
                                     anomaly_scores: Dict[str, float]) -> dict:
        """
        Create visualization data for multi-scale analysis
        """
        viz_data = {
            'decompositions': decompositions,
            'anomaly_scores': anomaly_scores,
            'scale_order': self.scales,
            'timestamps': np.arange(len(next(iter(decompositions.values()))))
        }
        
        return viz_data


class TemporalPatternMatcher:
    """
    Matches temporal patterns across scales to known attack signatures
    """
    
    def __init__(self):
        self.attack_signatures = self._load_attack_signatures()
        
    def _load_attack_signatures(self) -> Dict[str, Dict]:
        """Load known attack temporal signatures"""
        return {
            'ddos_syn_flood': {
                'scales': ['millisecond', 'second'],
                'patterns': {
                    'millisecond': {'peak_freq': 100, 'regularity': 0.9},
                    'second': {'burst_duration': 5, 'intensity': 0.8}
                }
            },
            'port_scan': {
                'scales': ['millisecond', 'second'],
                'patterns': {
                    'millisecond': {'interval': 50, 'variance': 0.1},
                    'second': {'sweep_rate': 100, 'coverage': 0.7}
                }
            },
            'data_exfiltration': {
                'scales': ['second', 'minute', 'hour'],
                'patterns': {
                    'second': {'transfer_rate': 1000, 'consistency': 0.8},
                    'minute': {'volume_trend': 'increasing', 'stealth': 0.6},
                    'hour': {'periodicity': 24, 'active_hours': [22, 23, 0, 1, 2]}
                }
            },
            'botnet_c2': {
                'scales': ['second', 'minute'],
                'patterns': {
                    'second': {'beacon_interval': 30, 'jitter': 0.2},
                    'minute': {'command_frequency': 2, 'response_time': 0.5}
                }
            }
        }
    
    def match_pattern(self, multi_scale_features: Dict[str, Dict]) -> Dict[str, float]:
        """
        Match observed patterns to known attack signatures
        """
        match_scores = {}
        
        for attack_type, signature in self.attack_signatures.items():
            score = 0
            matches = 0
            
            for scale in signature['scales']:
                if scale in multi_scale_features:
                    scale_match = self._match_scale_pattern(
                        multi_scale_features[scale],
                        signature['patterns'][scale]
                    )
                    score += scale_match
                    matches += 1
            
            if matches > 0:
                match_scores[attack_type] = score / matches
        
        return match_scores
    
    def _match_scale_pattern(self, observed: Dict, expected: Dict) -> float:
        """Match observed pattern to expected signature at specific scale"""
        matches = []
        
        for key, expected_value in expected.items():
            if key in observed:
                if isinstance(expected_value, (int, float)):
                    # Numerical match
                    diff = abs(observed[key] - expected_value) / (expected_value + 1e-6)
                    matches.append(1 - min(diff, 1))
                elif isinstance(expected_value, str):
                    # Categorical match
                    matches.append(1.0 if observed[key] == expected_value else 0.0)
                elif isinstance(expected_value, list):
                    # Set membership
                    matches.append(1.0 if observed[key] in expected_value else 0.0)
        
        return np.mean(matches) if matches else 0.0


def example_usage():
    """Demonstrate multi-scale temporal analysis"""
    
    # Create synthetic IoT data with multi-scale patterns
    sampling_rate = 10000  # 10kHz
    duration = 60  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Normal behavior: combination of multiple frequencies
    normal = (np.sin(2 * np.pi * 0.1 * t) +  # Slow variation (10s period)
              0.5 * np.sin(2 * np.pi * 1 * t) +  # 1Hz component
              0.2 * np.sin(2 * np.pi * 50 * t) +  # 50Hz component
              0.1 * np.random.randn(len(t)))  # Noise
    
    # Add multi-scale anomaly (DDoS-like pattern)
    anomaly_start = 30 * sampling_rate
    anomaly_end = 40 * sampling_rate
    
    # Microsecond scale: packet timing irregularities
    micro_anomaly = 0.5 * np.random.randn(anomaly_end - anomaly_start)
    
    # Millisecond scale: burst patterns
    milli_anomaly = 2 * signal.square(2 * np.pi * 100 * t[anomaly_start:anomaly_end] / sampling_rate)
    
    # Second scale: traffic surge
    second_anomaly = 3 * np.ones(anomaly_end - anomaly_start)
    
    # Combine anomalies
    data = normal.copy()
    data[anomaly_start:anomaly_end] += micro_anomaly + milli_anomaly + second_anomaly
    
    # Analyze with multi-scale analyzer
    analyzer = MultiScaleTemporalAnalyzer()
    
    # Decompose signal
    decompositions = analyzer.decompose_signal(data, sampling_rate)
    
    # Extract features at each scale
    features = analyzer.extract_scale_features(decompositions)
    
    # Detect anomalies
    anomaly_scores = analyzer.detect_scale_anomalies(features)
    
    # Fuse scores
    final_score = analyzer.fuse_multi_scale_scores(anomaly_scores, attack_type='ddos')
    
    print("Multi-Scale Temporal Analysis Results:")
    print("-" * 50)
    for scale, score in anomaly_scores.items():
        print(f"{scale.capitalize()} scale anomaly score: {score:.4f}")
    print(f"\nFused anomaly score: {final_score:.4f}")
    
    # Pattern matching
    matcher = TemporalPatternMatcher()
    attack_matches = matcher.match_pattern(features)
    
    print("\nAttack Pattern Matching:")
    print("-" * 50)
    for attack, confidence in sorted(attack_matches.items(), key=lambda x: x[1], reverse=True):
        print(f"{attack}: {confidence:.2%} confidence")
    
    return analyzer, decompositions, anomaly_scores


if __name__ == "__main__":
    analyzer, decompositions, scores = example_usage()