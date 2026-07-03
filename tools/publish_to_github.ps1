[CmdletBinding()]
param(
    [string]$RepositoryName = "fixed-bed-pyrolysis-trustworthiness",
    [string]$RemoteUrl = "",
    [string]$CommitMessage = "Prepare public reproducibility repository v1.0.0-rc4"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Install Git for Windows and reopen VS Code."
    }
}

Require-Command "git"
Write-Host "Repository root: $RepoRoot" -ForegroundColor Cyan

# Safety audit before any commit or push.
python .\tools\audit_public_repository.py
if ($LASTEXITCODE -ne 0) { throw "Public repository audit failed." }

# Ensure Git identity exists; prompt rather than inventing it.
$userName = git config --global user.name
$userEmail = git config --global user.email
if ([string]::IsNullOrWhiteSpace($userName)) {
    $userName = Read-Host "Enter the Git commit author name (recommended: Seyed Ali Shahnouri)"
    git config --global user.name $userName
}
if ([string]::IsNullOrWhiteSpace($userEmail)) {
    $userEmail = Read-Host "Enter the Git commit author email used by your GitHub account"
    git config --global user.email $userEmail
}

if (-not (Test-Path ".git")) {
    git init -b main
}

git add .
$status = git status --short
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "No uncommitted changes were found." -ForegroundColor Yellow
} else {
    git commit -m $CommitMessage
}

$hasOrigin = git remote 2>$null | Select-String -SimpleMatch "origin"
if ($RemoteUrl) {
    if ($hasOrigin) { git remote set-url origin $RemoteUrl }
    else { git remote add origin $RemoteUrl }
    git push -u origin main
    Write-Host "Pushed successfully to $RemoteUrl" -ForegroundColor Green
    exit 0
}

$gh = Get-Command "gh" -ErrorAction SilentlyContinue
if ($gh) {
    gh auth status
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GitHub CLI is installed but not authenticated. Run: gh auth login" -ForegroundColor Yellow
        exit 1
    }
    if (-not $hasOrigin) {
        gh repo create $RepositoryName --public --source . --remote origin --push `
          --description "Code and aggregate reproducibility package for trustworthiness-aware ML analysis of fixed-bed biomass pyrolysis yields."
    } else {
        git push -u origin main
    }
    Write-Host "GitHub publication completed." -ForegroundColor Green
    exit 0
}

Write-Host "GitHub CLI was not found. The local Git repository and first commit are ready." -ForegroundColor Yellow
Write-Host "Create an EMPTY public repository on GitHub named: $RepositoryName"
Write-Host "Do not add a README, .gitignore, or license on the GitHub creation page."
Write-Host "Then rerun this script with:"
Write-Host ".\tools\publish_to_github.ps1 -RemoteUrl https://github.com/YOUR_USERNAME/$RepositoryName.git" -ForegroundColor Cyan
