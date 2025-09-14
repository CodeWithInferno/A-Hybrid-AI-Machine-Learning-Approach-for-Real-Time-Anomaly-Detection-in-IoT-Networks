# PowerShell script to install LaTeX and compile the paper
# Run this script in PowerShell as Administrator

Write-Host "Installing MiKTeX (LaTeX distribution for Windows)..." -ForegroundColor Green

# Check if Chocolatey is installed
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey package manager..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Install MiKTeX using Chocolatey
Write-Host "Installing MiKTeX..." -ForegroundColor Yellow
choco install miktex -y

# Refresh environment variables
Write-Host "Refreshing environment variables..." -ForegroundColor Yellow
$env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Wait for installation to complete
Start-Sleep -Seconds 10

# Navigate to paper directory
Set-Location "C:\Users\prpatel.CSE-LAB\Documents\IOT\IOT Research\IOT\Paper"

Write-Host "Compiling LaTeX document..." -ForegroundColor Green

# Compile the LaTeX document (multiple passes for references)
try {
    Write-Host "First pass..." -ForegroundColor Cyan
    pdflatex -interaction=nonstopmode updated_conference_101719.tex
    
    Write-Host "Second pass (for references)..." -ForegroundColor Cyan
    pdflatex -interaction=nonstopmode updated_conference_101719.tex
    
    Write-Host "Third pass (final)..." -ForegroundColor Cyan
    pdflatex -interaction=nonstopmode updated_conference_101719.tex
    
    Write-Host "Success! PDF generated: updated_conference_101719.pdf" -ForegroundColor Green
    
    # Open the PDF
    if (Test-Path "updated_conference_101719.pdf") {
        Write-Host "Opening PDF..." -ForegroundColor Green
        Start-Process "updated_conference_101719.pdf"
    }
    
} catch {
    Write-Host "Error during compilation: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "You may need to restart PowerShell and try again after MiKTeX installation completes." -ForegroundColor Yellow
}

Write-Host "Done!" -ForegroundColor Green