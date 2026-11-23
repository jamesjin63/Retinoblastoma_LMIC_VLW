# Study inputs

The analysis reads three GBD batches, a registry table, a prior valuation table and a country boundary file from the input directory. The study's fixed reference series are included in `../reference/`. Obtain the GBD 2023 estimates from the [IHME Results Tool](https://vizhub.healthdata.org/gbd-results/) under its data-use terms. The study used Results Tool version 8352 and retinoblastoma cause ID 1009.

## GBD exports

Supply `R1.parquet`, `R2.parquet` and `POP.parquet`, or place non-overlapping CSV or CSV.GZ partitions directly inside directories named `R1/`, `R2/` and `POP/`. Partitions are read directly by `python run.py`. A Parquet file takes precedence over its corresponding directory.

| Batch | Location IDs | Sex IDs | Measure IDs | Metric IDs | Rows |
|---|---|---|---|---|---:|
| R1 | 130 IDs in location_crosswalk.csv | 1, 2, 3 | 1–6 | 1, 3 | 318,240 |
| R2 | 1, 5, 159, 174, 199 | 3 | 1–6 | 1, 3 | 4,080 |
| POP | Same as R1 | 1, 2, 3 | 44 | 1 | 26,520 |

Use every year from 1990 through 2023, age IDs 1 and 6, and the all-population group. R1 includes DALYs for every year. For R1 and R2, measures 1–6 are deaths, DALYs, YLDs, YLLs, prevalence and incidence respectively. Measure 44 is population. Sex IDs 1–3 are male, female and both sexes. Metric 1 is number and metric 3 is rate per 100,000. R2 covers the globe, East Asia, South Asia, Eastern sub-Saharan Africa and Western sub-Saharan Africa.

The unique key is `location, year, age, sex, measure, metric`. Include numeric `val, lower, upper` and the disease `cause` for R1 and R2. ID columns can also use the suffix `_id`. If present, `population_group` must equal 1. Each batch must cover its full grid with finite nonnegative estimates and ordered bounds.

To save a validated set of CSV partitions as Parquet, run this command from the repository root before placing any Parquet files in the input directory. The converter accepts the same numeric ID fields as the analysis entry point.

```sh
python - <<'PYTHON'
from pathlib import Path
import pandas as pd
from inputs import read_batch

folder = Path("data")
countries = pd.read_csv("reference/location_crosswalk.csv").location_id.tolist()
for name, locations in [("R1", countries), ("R2", [1, 5, 159, 174, 199]), ("POP", countries)]:
    destination = folder / f"{name}.parquet"
    if destination.exists():
        raise FileExistsError(destination)
    batch = read_batch(folder, name, locations)
    if name != "POP":
        batch["cause"] = 1009
    batch.to_parquet(destination, index=False)
PYTHON
```

## Additional files in the input directory

| Filename | Source and required content |
|---|---|
| 05-retinoblastoma.csv | Retinoblastoma diagnostic group table from [International Incidence of Childhood Cancer, volume III](https://iicc.iarc.who.int/). One record per registry population and observation period. Include country_or_territory, registry_population_label, observation_period (start-end), cases_0_14, rate_0_4_per_million and rate_5_9_per_million. Keep published registry labels and uppercase country names. |
| Table_10_Retinoblastoma.csv | Country table from the prior 0–19-year valuation. Request the study extract from the authors for the exact bridge comparison. Two introductory lines precede the CSV header. Required fields are location_id (unique), DALY (years), VSL_IntUSD (2023 international dollars), HALE_years and VLW_nondisc_millionIntUSD (millions of 2023 international dollars). |
| df_world2.geojson | Study boundary snapshot with a declared coordinate reference system, polygon geometries, brk_a3 and numeric GBD location_id. Request the study geometry for exact map replication. The China outline uses the [DataV national boundary](https://geo.datav.aliyun.com/areas_v3/bound/100000.json). |

## Bundled reference files

| Filename | Key and variables used |
|---|---|
| location_crosswalk.csv | One row per location_id. Country name, ISO3, fixed World Bank FY2027 income group and GBD region. Groups are Low income, Lower middle income and Upper middle income. |
| age_weights_gbd2023_reconstructed.csv | One row per age_id. The two gbd2023_all_age_weight values for age IDs 1 and 6 are renormalised at runtime. |
| wdi_gni_ppp_current.csv | ISO3 and year. Per-capita GNI in current international dollars, NY.GNP.PCAP.PP.CD. |
| wdi_us_deflator.csv | One US value per year, GDP deflator NY.GDP.DEFL.ZS. |
| wdi_macro_long_1990_2023.csv | ISO3, year and indicator. Per-capita GDP in current USD (gdppc_current_us, NY.GDP.PCAP.CD), current international dollars (gdppc_ppp_current_intl, NY.GDP.PCAP.PP.CD), and population in persons (population_total, SP.POP.TOTL). |
| wpp_life_table_both_ex.csv.gz | ISO3, year and age_x. Both-sex remaining life expectancy in years. |
| wpp_median_age.csv | ISO3 and year. National median_age in years. |
| k_threshold_130.csv | One row per location_id. k_main is the Pichon-Riviere/Ochalek opportunity-cost multiplier applied to per-capita GDP. k_main_source records the country-specific source or fallback. |
| lit_rb_cost.csv | Five estimates from four published studies, identified by study and cost_type. Includes currency, reported value per patient, sample size, study year, reported or assumed price year and extraction notes. |

Economic series come from the [World Development Indicators](https://databank.worldbank.org/source/world-development-indicators) and demographic series from [UN World Population Prospects 2024](https://population.un.org/wpp/). The analysis uses 1990–2023 and includes the US income reference. Country-level missing GNI is first estimated from GDP and the country's median observed GNI/GDP ratio. Remaining internal gaps use log-linear interpolation, leading gaps use a flat real backcast, and a single missing final year is carried forward. Supplementary Table 13 reports coverage.
