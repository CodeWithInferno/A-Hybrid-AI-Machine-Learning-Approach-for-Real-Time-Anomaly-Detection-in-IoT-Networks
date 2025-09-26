# LaTeX Setup Guide for Windows

## Quick Setup Options

### Option 1: MiKTeX (Recommended for Windows)
1. Download MiKTeX from: https://miktex.org/download
2. Run the installer (choose "Install for all users")
3. Let it install (takes 10-15 minutes)
4. MiKTeX will auto-download packages as needed

### Option 2: TeX Live (Full Installation)
1. Download from: https://tug.org/texlive/acquire-netinstall.html
2. Run install-tl-windows.exe
3. Choose "Full scheme" (4GB download)
4. Wait for installation (30-45 minutes)

## Editor Options

### 1. TeXstudio (Easiest)
- Download: https://www.texstudio.org/
- Auto-detects MiKTeX/TeX Live
- Built-in PDF viewer
- Syntax highlighting

### 2. VS Code with LaTeX Workshop
- Install VS Code
- Add "LaTeX Workshop" extension
- Modern interface
- Git integration

### 3. Overleaf (Online - No Installation)
- Go to: https://www.overleaf.com
- Create free account
- Upload your `New_Approach/paper/` folder
- Compile online

## To Compile Your Paper

### Using TeXstudio:
1. Open `main_extended.tex`
2. Press F5 (or click green arrow)
3. PDF appears on the right

### Using Command Line:
```bash
cd "C:\Users\prpatel.CSE-LAB\Documents\IOT\IOT Research\IOT\New_Approach\paper"
pdflatex main_extended.tex
bibtex main_extended
pdflatex main_extended.tex
pdflatex main_extended.tex
```

### Common Issues:

1. **Missing packages**: MiKTeX will auto-install, just click "Install" when prompted

2. **Bibliography not showing**: Run the compilation sequence above (pdflatex → bibtex → pdflatex → pdflatex)

3. **Figures not found**: Make sure you're compiling from the `paper/` directory

## Your Paper is Ready!

Your `main_extended.tex` has:
- All mathematical formulations
- 97.3% accuracy results
- Game-theoretic proofs
- Multi-scale algorithms
- 10 pages of content

Just compile and submit! 🎉