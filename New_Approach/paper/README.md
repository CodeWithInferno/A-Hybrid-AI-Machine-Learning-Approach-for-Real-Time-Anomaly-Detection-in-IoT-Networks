# Temporal Deception Detection in IoT Networks - Paper

This directory contains the LaTeX source files for the paper "Temporal Deception Detection in IoT Networks: A Game-Theoretic Multi-Scale Hybrid Approach".

## Structure

- `main.tex` - Main LaTeX document (IEEE two-column format)
- `bibliography.bib` - Bibliography file with all references
- `figures/` - Directory for figure files
- `Makefile` - Build automation

## Building the Paper

### Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- pdflatex
- bibtex

### Compilation

#### Using Make (Linux/Mac):
```bash
make
```

#### Using pdflatex directly:
```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

#### Using latexmk:
```bash
latexmk -pdf main
```

### Clean build files:
```bash
make clean
```

### Remove all generated files including PDF:
```bash
make distclean
```

## Paper Highlights

- **8-10 pages** following IEEE conference format
- **Mathematical rigor** with theorems, proofs, and formal definitions
- **Comprehensive experiments** on real-world IoT botnet datasets
- **Novel contributions**:
  - Fundamental Theorem of Temporal Signatures
  - Multi-scale hybrid LSTM+ARIMA architecture
  - Game-theoretic fusion mechanism
  - Convergence and complexity guarantees

## Key Sections

1. **Introduction**: Problem motivation and contributions
2. **Related Work**: Survey of IoT security approaches
3. **Problem Formulation**: Mathematical framework
4. **Theoretical Foundation**: Core theorems and proofs
5. **Proposed Framework**: System architecture and algorithms
6. **Experimental Evaluation**: Comprehensive results
7. **Discussion**: Implications and deployment
8. **Conclusion**: Summary and future work

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{temporal_deception_2025,
  title={Temporal Deception Detection in IoT Networks: A Game-Theoretic Multi-Scale Hybrid Approach},
  author={Anonymous Authors},
  booktitle={Proceedings of IEEE S&P/NDSS/ACM CCS},
  year={2025}
}
```

## Notes for Authors

- Update author information in `main.tex` before submission
- Ensure all figures are in `figures/` directory
- Run spell check and grammar check
- Verify references are complete and accurate
- Check that the paper compiles without errors or warnings