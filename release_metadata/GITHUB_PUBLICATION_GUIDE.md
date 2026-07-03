# GitHub publication guide

## Recommended repository settings

- **Repository name:** `fixed-bed-pyrolysis-trustworthiness`
- **Visibility:** Public
- **Description:** Code and aggregate reproducibility package for trustworthiness-aware ML analysis of fixed-bed biomass pyrolysis yields.
- **Default branch:** `main`
- **Topics:** `biomass-pyrolysis`, `machine-learning`, `grouped-validation`, `conformal-prediction`, `applicability-domain`, `reproducible-research`

## Recommended Windows path

Extract this ZIP to a clean directory, for example:

```text
E:\fixed-bed-pyrolysis-trustworthiness
```

Do not extract it inside the analytical working directory containing the restricted row-level files.

## Preferred publication method

Open the extracted folder in VS Code, then in PowerShell run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\tools\publish_to_github.ps1
```

The script:

1. checks that restricted files are absent;
2. asks for Git identity if it is not configured;
3. initializes the `main` branch;
4. creates the first commit;
5. uses authenticated GitHub CLI when available; otherwise it prints the exact fallback command.

## Manual fallback

Create an **empty** public repository on GitHub. Do not initialize it with a README, license, or `.gitignore`, because those files already exist locally. Then run:

```powershell
.\tools\publish_to_github.ps1 -RemoteUrl https://github.com/YOUR_USERNAME/fixed-bed-pyrolysis-trustworthiness.git
```

## After the first push

- verify the green Repository audit workflow;
- add the recommended repository topics;
- do not create `v1.0.0` yet;
- connect the repository to Zenodo;
- reserve/create the restricted record first;
- insert both DOI values and the GitHub URL into the metadata;
- publish the final GitHub release as `v1.0.0` only after those edits.

## Why no `.zenodo.json` is included yet

This repository uses `CITATION.cff`. A `.zenodo.json` file would override it during GitHub release archiving. Because the deposit has mixed licensing and still needs the companion restricted-record relation, final Zenodo metadata should be checked in the Zenodo interface after the repository is connected.

`CITATION.cff` identifies MIT as the primary software license for maximum GitHub–Zenodo compatibility. The CC BY 4.0 terms for author-generated figures, tables, aggregate outputs, and documentation remain explicitly defined in `LICENSES.md` and should be added as the second license in the Zenodo deposit interface.
