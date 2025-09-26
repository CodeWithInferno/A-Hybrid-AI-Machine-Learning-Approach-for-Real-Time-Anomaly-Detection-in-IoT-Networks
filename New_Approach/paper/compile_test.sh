#!/bin/bash

echo "Testing LaTeX compilation..."
echo

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Please install a LaTeX distribution."
    echo "Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "Mac: brew install mactex"
    echo "Or visit: https://www.latex-project.org/get/"
    exit 1
fi

echo "Found pdflatex. Starting compilation..."
echo

# First pass
echo "Running first pass..."
pdflatex -interaction=nonstopmode main.tex
if [ $? -ne 0 ]; then
    echo "ERROR: First LaTeX compilation pass failed."
    echo "Check main.log for details."
    exit 1
fi

# BibTeX
echo "Running BibTeX..."
bibtex main || echo "WARNING: BibTeX failed. Continuing without bibliography."

# Second pass
echo "Running second pass..."
pdflatex -interaction=nonstopmode main.tex

# Third pass (for references)
echo "Running third pass..."
pdflatex -interaction=nonstopmode main.tex

echo
echo "Compilation completed!"
echo "Output: main.pdf"

# Check if PDF was created
if [ -f main.pdf ]; then
    echo "SUCCESS: PDF generated successfully."
else
    echo "ERROR: PDF was not generated."
fi

echo
echo "Cleaning auxiliary files..."
rm -f *.aux *.log *.bbl *.blg *.out *.toc 2>/dev/null

echo "Done!"