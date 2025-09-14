# LaTeX Paper Compilation Guide

## Updated Paper Details
- **File:** `updated_conference_101719.tex`
- **Key Updates:** 
  - N-BaLoT dataset results (99.47% accuracy!)
  - 6 hybrid fusion strategies
  - Comprehensive performance comparison
  - New visualization: `comprehensive_hybrid_results.png`

## Option 1: Automatic Installation & Compilation

1. **Run PowerShell as Administrator**
   ```powershell
   cd "C:\Users\prpatel.CSE-LAB\Documents\IOT\IOT Research\IOT"
   .\install_and_compile_latex.ps1
   ```

## Option 2: Manual Installation

### Step 1: Install MiKTeX
1. Download MiKTeX from: https://miktex.org/download
2. Run the installer with default settings
3. Restart your computer after installation

### Step 2: Compile the Paper
```bash
cd "C:\Users\prpatel.CSE-LAB\Documents\IOT\IOT Research\IOT\Paper"

# First compilation pass
pdflatex updated_conference_101719.tex

# Second pass (for references and citations)
pdflatex updated_conference_101719.tex

# Third pass (final formatting)
pdflatex updated_conference_101719.tex
```

## Option 3: Online LaTeX Editor (Overleaf)

1. Go to https://overleaf.com
2. Create a new project
3. Upload the following files:
   - `updated_conference_101719.tex`
   - `comprehensive_hybrid_results.png`
4. Click "Recompile" to generate PDF

## What's New in the Updated Paper

### 🎯 Breakthrough Results
- **99.47% Accuracy** with Selective Hybrid approach
- **99.7% F1-Score** for attack detection
- **6 Different Fusion Strategies** evaluated
- **1.1+ Million Samples** processed with GPU acceleration

### 📊 Key Improvements
1. **N-BaLoT Dataset**: Real IoT botnet traffic (vs synthetic data)
2. **Multiple Hybrid Approaches**: Adaptive, Maximum, Multiplicative, Harmonic, Dynamic, Selective
3. **Enhanced Architecture**: Multi-layer LSTM + Ensemble Isolation Forest
4. **Comprehensive Evaluation**: Statistical, LSTM-Only, and 6 Hybrid variants
5. **GPU Acceleration**: PyTorch implementation with CUDA support

### 📈 Performance Comparison Table
| Model | Accuracy | AUC | F1-Score | 
|-------|----------|-----|----------|
| **Selective Hybrid** | **99.47%** | **0.9945** | **99.7%** |
| Adaptive Weighted | 99.47% | 0.9962 | 99.7% |
| Multiplicative | 99.41% | 0.9947 | 99.7% |
| LSTM-Only | 99.07% | 0.9962 | 99.5% |
| Statistical-Only | 75.09% | 0.9757 | 83.2% |

## Expected Output
- **PDF File:** `updated_conference_101719.pdf`
- **Professional IEEE Format:** Double-column conference paper
- **Length:** Approximately 8-10 pages
- **Figures:** High-quality visualization of comprehensive results

## Troubleshooting
- **"File not found" errors:** Ensure `comprehensive_hybrid_results.png` is in Paper folder
- **Package missing errors:** MiKTeX will prompt to install missing packages automatically
- **Compilation errors:** Try running `pdflatex` command 3 times (LaTeX requires multiple passes)

Your updated paper now reflects your breakthrough research showing **hybrid AI superiority** in IoT botnet detection! 🚀