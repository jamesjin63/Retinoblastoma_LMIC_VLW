# Retinoblastoma burden and value of lost welfare in children under ten

This study estimates health loss from retinoblastoma in 130 low- and middle-income countries from 1990 to 2023. Monetary estimates cover the 124 countries with the required income and GDP data. The main analysis reports undiscounted value of lost welfare in 2023 US dollars.

## Article figures and tables

| Collection | Files |
|---|---|
| [Main figures](figures/) | Figures 1–4 (PDF) |
| [Supplementary figures](supplementary_figures/) | Figures S1–S5 (PDF) |
| [Main tables](tables/) | Tables 1–2 (CSV) |
| [Supplementary tables](supplementary_tables/) | Tables S1–S14 (CSV) |

## Run the analysis

Use Python 3.10 and run these commands from the repository directory. Before running the analysis, place the six study inputs in `data/` as described in the [input guide](data/README.md).

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

To use another input or output directory, run `python run.py --data-dir /path/to/inputs --output-dir /path/to/results`. The fixed economic, demographic and cohort inputs are read from `reference/`.

## Generated outputs

Running the analysis writes figures in PDF and PNG, tables in CSV and Markdown, and figure source data in CSV and Excel to `results/`. Figure workbooks have one worksheet per panel. The combined CSV identifies the panel for each record.
