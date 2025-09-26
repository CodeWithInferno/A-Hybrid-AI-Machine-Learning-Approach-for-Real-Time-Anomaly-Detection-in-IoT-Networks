@echo off
echo Testing LaTeX compilation...
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: pdflatex not found. Please install a LaTeX distribution.
    echo Visit: https://www.latex-project.org/get/
    exit /b 1
)

echo Found pdflatex. Starting compilation...
echo.

REM First pass
echo Running first pass...
pdflatex -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo ERROR: First LaTeX compilation pass failed.
    echo Check main.log for details.
    exit /b 1
)

REM BibTeX
echo Running BibTeX...
bibtex main
if %errorlevel% neq 0 (
    echo WARNING: BibTeX failed. Continuing without bibliography.
)

REM Second pass
echo Running second pass...
pdflatex -interaction=nonstopmode main.tex

REM Third pass (for references)
echo Running third pass...
pdflatex -interaction=nonstopmode main.tex

echo.
echo Compilation completed!
echo Output: main.pdf

REM Check if PDF was created
if exist main.pdf (
    echo SUCCESS: PDF generated successfully.
) else (
    echo ERROR: PDF was not generated.
)

echo.
echo Cleaning auxiliary files...
del /q *.aux *.log *.bbl *.blg *.out *.toc 2>nul

pause