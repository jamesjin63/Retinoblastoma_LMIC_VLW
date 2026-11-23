#!/usr/bin/env python3
"""Estimate retinoblastoma burden and lost welfare in children under ten."""

from pathlib import Path
from inputs import analysis_paths, read_batch

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr

DATA, OUT = analysis_paths()
EXT = Path(__file__).resolve().parent / "reference"
FIG, FIG_S = OUT / "figures", OUT / "supplementary_figures"
TAB_MAIN, TAB_S = OUT / "tables", OUT / "supplementary_tables"
SD = OUT / "source_data"
for directory in (FIG, FIG_S, TAB_MAIN, TAB_S, SD):
    directory.mkdir(parents=True, exist_ok=True)


ND = 1000
RNG = np.random.default_rng(20261123)
YEARS = np.arange(1990, 2024)
NY = len(YEARS)
Y23 = NY - 1
MEAS = {1: "Deaths", 2: "DALYs", 3: "YLDs", 4: "YLLs", 5: "Prevalence", 6: "Incidence"}
MI = {m: i for i, m in enumerate(MEAS)}
AGES = [1, 6]
AGE_LAB = {1: "<5", 6: "5–9"}
SEXI = {1: 0, 2: 1, 3: 2}
INC = ["Low income", "Lower middle income", "Upper middle income"]
INC_SHORT = {"Low income": "LIC", "Lower middle income": "LMC", "Upper middle income": "UMC"}
INC_COL = {"Low income": "#C44E52", "Lower middle income": "#DD8452", "Upper middle income": "#4C72B0",
           "All LMICs": "#000000"}
FOCUS = ["Eastern Sub-Saharan Africa", "Western Sub-Saharan Africa", "South Asia", "East Asia"]
REGION_LID = {"Global": 1, "East Asia": 5, "South Asia": 159, "Eastern Sub-Saharan Africa": 174,
              "Western Sub-Saharan Africa": 199}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .8, "lines.linewidth": 1.6,
})


def panel(ax, k, x=-0.12, fs=13):
    ax.text(x, 1.02, f"({'abcdefgh'[k]})", transform=ax.transAxes, fontweight="bold", va="bottom", ha="left",
            fontsize=fs)


def save(fig, name, *, layout_ready=False):
    folder = FIG_S if name.startswith("SuppFig") else FIG
    if not layout_ready:
        fig.tight_layout()
    fig.savefig(folder / f"{name}.png", bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt_int(x):
    return "—" if pd.isna(x) else f"{int(round(x)):,}".replace("-", "−")


def fmt_dec(x, k=1, sign=False):
    if pd.isna(x):
        return "—"
    s = f"{x:+,.{k}f}" if sign else f"{x:,.{k}f}"
    return s.replace("-", "−")


def fmt_ui(p, lo, hi, k=None):
    f = fmt_int if k is None else (lambda v: fmt_dec(v, k))
    return "—" if pd.isna(p) else f"{f(p)} ({f(lo)} to {f(hi)})"


def q(a, axis=0):
    return np.nanpercentile(a, 2.5, axis=axis), np.nanpercentile(a, 97.5, axis=axis)


def require_agreement(check, detail, diff, tol=0.0):
    if not np.isfinite(diff) or abs(diff) > tol:
        raise ValueError(f"{check} ({detail}): difference {diff}, tolerance {tol}")


# Cohort and geography
cw = pd.read_csv(EXT / "location_crosswalk.csv").sort_values("location_id").reset_index(drop=True)
if len(cw) != 130 or cw.location_id.duplicated().any() or cw.iso3.duplicated().any():
    raise ValueError("Expected 130 distinct cohort locations and ISO codes.")
LIDS = cw.location_id.tolist()
LIDX = {l: i for i, l in enumerate(LIDS)}
ISO = cw.iso3.tolist()
NC = len(LIDS)
NAME = dict(zip(cw.location_id, cw.location_name))
INCG = cw.LMIC_group.values
REGN = cw.region_name.values
GROUPS = {"All LMICs": np.ones(NC, bool)}
GROUPS.update({g: INCG == g for g in INC})
GROUPS.update({r: REGN == r for r in FOCUS})
GN = list(GROUPS)
GMAIN = ["All LMICs"] + INC


# GBD data (R1, R2, POP)
r1 = read_batch(DATA, "R1", LIDS)


# cube arrays: [measure, sex, age, country, year]
SHAPE = (6, 3, 2, NC, NY)


def cube(df, metric):
    arr = {k: np.full(SHAPE, np.nan) for k in ("val", "lower", "upper")}
    d = df[df.metric == metric]
    ii = (d.measure.map(MI).values, d.sex.map(SEXI).values, d.age.map({1: 0, 6: 1}).values,
          d.location.map(LIDX).values, (d.year - 1990).values)
    for k in arr:
        arr[k][ii] = d[k].values
    return arr


N = cube(r1, 1)
R = cube(r1, 3)
missing_cells = int(np.isnan(N["val"]).sum())
require_agreement("Number cube complete", "6 × 3 × 2 × 130 × 34", missing_cells)

# population: GBD download where present; otherwise mean Number / Rate across available measures × 1e5
pop_raw = read_batch(DATA, "POP", LIDS)
POP = np.full((3, 2, NC, NY), np.nan)
with np.errstate(divide="ignore", invalid="ignore"):
    POP_D = np.nanmean(np.where(R["val"] > 0, N["val"] / R["val"] * 1e5, np.nan), axis=0)
if len(pop_raw):
    p = pop_raw[pop_raw.location.isin(LIDS) & pop_raw.age.isin(AGES)]
    POP[p.sex.map(SEXI).values, p.age.map({1: 0, 6: 1}).values, p.location.map(LIDX).values,
        (p.year - 1990).values] = p.val.values
    ok = ~np.isnan(POP) & ~np.isnan(POP_D)
    require_agreement("GBD population vs Number/Rate-derived population (max rel. diff)", f"{ok.sum():,} cells",
       float(np.nanmax(np.abs(POP[ok] / POP_D[ok] - 1))) if ok.any() else np.nan, 0.01)
POP_SRC = np.where(np.isnan(POP), "derived (Number/Rate)", "GBD population")
POP = np.where(np.isnan(POP), POP_D, POP)
require_agreement("Population available for every country-year-sex-age", "<5, 5–9", int(np.isnan(POP).sum()))
N_POP_DERIVED = int((POP_SRC[2] == "derived (Number/Rate)").sum())

# consistency checks
v = N["val"]
require_agreement("Males + females = both sexes (max rel. diff, all measures)", "Number",
   float(np.nanmax(np.abs((v[:, 0] + v[:, 1]) / np.where(v[:, 2] > 0, v[:, 2], np.nan) - 1))), 0.01)
require_agreement("DALY = YLL + YLD (max rel. diff)", "both sexes, Number",
   float(np.nanmax(np.abs((v[MI[3], 2] + v[MI[4], 2]) / v[MI[2], 2] - 1))), 0.01)
with np.errstate(divide="ignore", invalid="ignore"):
    rr = N["val"][:, :, :, :, :] / POP[None] * 1e5 / R["val"]
require_agreement("Rate = Number / Population × 1e5 (median abs. rel. diff)", "all cells with rate > 0",
   float(np.nanmedian(np.abs(rr[R["val"] > 0] - 1))), 0.01)

# GBD 2023 standard-population weights renormalised within <10 years
aw = pd.read_csv(EXT / "age_weights_gbd2023_reconstructed.csv").set_index("age_id").gbd2023_all_age_weight
W = np.array([aw[1], aw[6]])
W = W / W.sum()


# Monte Carlo draws (split lognormal)
def split_lognormal(val, lo, hi, z):
    """val/lo/hi broadcast against z (draw axis first). Median = val; 2.5/97.5% ≈ lo/hi."""
    val = np.asarray(val, float)
    lo = np.minimum(np.maximum(lo, val * 1e-4), val)
    hi = np.maximum(hi, val)
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = np.where(val > 0, (np.log(val) - np.log(np.where(lo > 0, lo, 1))) / 1.96, 0)
        su = np.where(val > 0, (np.log(np.where(hi > 0, hi, 1)) - np.log(np.where(val > 0, val, 1))) / 1.96, 0)
    out = val * np.exp(np.where(z > 0, z * su, z * sl))
    return np.where(val > 0, out, 0.0)


Z_IND = RNG.standard_normal((ND, NC))             # independent between countries (main)
Z_COR = np.repeat(RNG.standard_normal((ND, 1)), NC, axis=1)   # fully correlated between countries


def draws(m, sex=2, years=slice(None), z=Z_IND):
    """Draws of Number for measure m: (ND, 2 ages, NC, years). Same z across ages, measures, years in a country."""
    zz = z[:, None, :, None]
    return split_lognormal(N["val"][MI[m], sex][None, :, :, years], N["lower"][MI[m], sex][None, :, :, years],
                           N["upper"][MI[m], sex][None, :, :, years], zz)


D_DALY = draws(2)                                  # (ND, 2, NC, NY)
D23 = {m: draws(m, years=slice(Y23, NY))[..., 0] for m in MEAS}   # (ND, 2, NC)
D23_COR = {m: draws(m, years=slice(Y23, NY), z=Z_COR)[..., 0] for m in MEAS}

# QC of the simulation: per-cell 2.5/97.5% vs GBD bounds (2023, both sexes, DALY)
lo_s, hi_s = q(D23[2])
gl, gh = N["lower"][MI[2], 2, :, :, Y23], N["upper"][MI[2], 2, :, :, Y23]
ok = gl > 0
require_agreement("Simulated 2.5% vs GBD lower (median abs. rel. diff)", "DALY 2023 cells", float(np.median(np.abs(lo_s[ok] / gl[ok] - 1))), .05)
require_agreement("Simulated 97.5% vs GBD upper (median abs. rel. diff)", "DALY 2023 cells", float(np.median(np.abs(hi_s[ok] / gh[ok] - 1))), .05)
SIMQC = pd.DataFrame({"location_id": np.repeat(LIDS, 2), "Country": np.repeat([NAME[l] for l in LIDS], 2),
                      "Age": ["<5", "5–9"] * NC,
                      "GBD DALY": N["val"][MI[2], 2, :, :, Y23].T.ravel(),
                      "GBD lower": gl.T.ravel(), "GBD upper": gh.T.ravel(),
                      "Simulated median": np.median(D23[2], axis=0).T.ravel(),
                      "Simulated 2.5%": lo_s.T.ravel(), "Simulated 97.5%": hi_s.T.ravel()})

# Income, life tables, VSL, VSLY
macro = pd.read_csv(EXT / "wdi_macro_long_1990_2023.csv")
gni_raw = pd.read_csv(EXT / "wdi_gni_ppp_current.csv")
usdef = pd.read_csv(EXT / "wdi_us_deflator.csv").set_index("year").value
DEFL = (usdef.loc[2023] / usdef.loc[YEARS]).values      # current → 2023 prices


def wide(df, ind=None):
    d = df if ind is None else df[df.indicator == ind]
    d = d[d.value.notna() & d.year.between(1990, 2023)]
    return d.pivot_table(index="iso3", columns="year", values="value").reindex(index=ISO + ["USA"], columns=YEARS)


GNI = wide(gni_raw) * DEFL
GDP = wide(macro, "gdppc_ppp_current_intl") * DEFL
GDPUSD = wide(macro, "gdppc_current_us")
GDPPPP_CUR = wide(macro, "gdppc_ppp_current_intl")
POPTOT = wide(macro, "population_total")
FLAG = pd.DataFrame(np.where(GNI.notna(), "observed", ""), index=GNI.index, columns=YEARS)
ratio = (GNI / GDP).median(axis=1)
m2 = GNI.isna() & GDP.notna() & ratio.notna().values[:, None]
GNI = GNI.mask(m2, GDP.mul(ratio, axis=0))
FLAG = FLAG.mask(m2, "GDP × GNI/GDP ratio")
GDP = GDP.mask(GDP.isna() & GNI.notna(), GNI.div(ratio, axis=0))


def fill_row(v, flags=None):
    v = v.copy()
    ok = np.where(~np.isnan(v))[0]
    if len(ok) == 0:
        return v, flags
    first, last = ok[0], ok[-1]
    lv = np.log(v)
    for i in range(first, last + 1):
        if np.isnan(v[i]):
            v[i] = np.exp(np.interp(i, ok, lv[ok]))
            if flags is not None:
                flags[i] = "interpolated (log-linear)"
    for i in range(first):
        v[i] = v[first]
        if flags is not None:
            flags[i] = "backcast (flat, real terms)"
    if last == len(v) - 2:
        v[-1] = v[last]
        if flags is not None:
            flags[-1] = "carried forward 1 year"
    return v, flags


gni_f, gdp_f, flag_f, pop_f = [], [], [], []
for iso in GNI.index:
    fl = FLAG.loc[iso].values.astype(object)
    a, fl = fill_row(GNI.loc[iso].values.astype(float), fl)
    b, _ = fill_row(GDP.loc[iso].values.astype(float))
    c, _ = fill_row(POPTOT.loc[iso].values.astype(float))
    gni_f.append(a), gdp_f.append(b), flag_f.append(fl), pop_f.append(c)
GNI = pd.DataFrame(gni_f, index=GNI.index, columns=YEARS)
GDP = pd.DataFrame(gdp_f, index=GDP.index, columns=YEARS)
POPTOT = pd.DataFrame(pop_f, index=POPTOT.index, columns=YEARS)
FLAG = pd.DataFrame(flag_f, index=FLAG.index, columns=YEARS).replace("", "missing")
G = GNI.loc[ISO].values                       # (NC, NY) 2023 I$ (PPP), real
GUS = GNI.loc["USA"].values
# Reporting currency: 2023 US$ at market exchange rates. VSL is transferred on PPP income
# (Robinson 2019); each country's amounts are then converted with its 2023 price-level ratio
# XR = GDP pc (current US$) / GDP pc (current PPP), held fixed over years.
XR = (GDPUSD.loc[ISO] / GDPPPP_CUR.loc[ISO]).ffill(axis=1)[2023].values   # US$ per I$
GDPPC = GDP.loc[ISO].values * XR[:, None]     # GDP per capita, 2023 US$ (real, US-deflated)
GDPTOT = GDPPC * POPTOT.loc[ISO].values       # total GDP, 2023 US$
MON = ~np.isnan(G[:, -1]) & ~np.isnan(GDPTOT[:, -1]) & ~np.isnan(XR)
MONH = MON & ~np.isnan(G).any(axis=1)         # full 1990–2023 series (trend sample)
OBS23 = (FLAG.loc[ISO][2023].values == "observed")
require_agreement("US GNI per capita complete 1990–2023", "USA", int(np.isnan(GUS).sum()))
require_agreement("US$ conversion factor available for monetary set", f"{MON.sum()} countries", int(np.isnan(XR[MON]).sum()))
require_agreement("Monetary set has full 1990–2023 income series", f"{MON.sum()} countries", int((MON & ~MONH).sum()))

ex = pd.read_csv(EXT / "wpp_life_table_both_ex.csv.gz")
ex = ex[ex.iso3.isin(ISO) & ex.year.between(1990, 2023)]
med = pd.read_csv(EXT / "wpp_median_age.csv").pivot_table(index="iso3", columns="year", values="median_age")
EMED = np.full((NC, NY), np.nan)
for (iso, yr), g in ex.groupby(["iso3", "year"]):
    g = g.sort_values("age_x")
    EMED[ISO.index(iso), yr - 1990] = np.interp(med.loc[iso, yr], g.age_x.values, g.remaining_years.values)
require_agreement("e(median age) available", "130 × 1990–2023", int(np.isnan(EMED).sum()))


def vsl(g, gus, eps=1.5, ratio_us=160.0, floor=20.0):
    core = ratio_us * gus * (g / gus) ** eps
    return core if floor is None else np.maximum(floor * g, core)


def annuity(e, r=0.03):
    v = 1 / (1 + r)
    return (1 - v ** e) / (1 - v)


VSL_PPP = vsl(G, GUS[None, :])                # 2023 I$ (PPP)
VSL = VSL_PPP * XR[:, None]                   # 2023 US$
VSLY = VSL / EMED
FLOOR = 20 * G >= 160 * GUS[None, :] * (G / GUS[None, :]) ** 1.5
VSLY23 = VSLY[:, -1]

# 2023 country results
u10 = lambda a: a.sum(axis=-2) if a.ndim == 3 else a.sum(axis=0)   # sum over age axis


def country_frame():
    rows = []
    pop23 = POP[2, :, :, Y23].sum(axis=0)
    for i, lid in enumerate(LIDS):
        rows.append({"location_id": lid, "iso3": ISO[i], "Country": NAME[lid], "Income group": INCG[i],
                     "Region": REGN[i]})
    c = pd.DataFrame(rows)
    c["Children <10"] = pop23
    for m, lab in MEAS.items():
        c[lab] = N["val"][MI[m], 2, :, :, Y23].sum(axis=0)
        lo, hi = q(D23[m].sum(axis=1))
        c[lab + " lower"], c[lab + " upper"] = lo, hi
    # age-standardised <10 rates per 100,000
    for m, lab in MEAS.items():
        rate = N["val"][MI[m], 2, :, :, Y23] / POP[2, :, :, Y23] * 1e5          # (2, NC)
        c[lab + " ASR"] = (W[:, None] * rate).sum(axis=0)
        rd = D23[m] / POP[2, :, :, Y23][None] * 1e5
        c[lab + " ASR lower"], c[lab + " ASR upper"] = q((W[None, :, None] * rd).sum(axis=1))
    c["Share <5 of DALYs, %"] = N["val"][MI[2], 2, 0, :, Y23] / c["DALYs"] * 100
    c["YLL share of DALYs, %"] = c["YLLs"] / c["DALYs"] * 100
    c["GNI pc, I$"] = G[:, -1]
    c["GNI source 2023"] = FLAG.loc[ISO][2023].values
    c["VSL floor binds"] = FLOOR[:, -1]
    c["e(median age)"] = EMED[:, -1]
    c["VSLY, US$"] = VSLY23
    dv = D23[2].sum(axis=1) * VSLY23[None]
    c["VLW, US$ million"] = c["DALYs"] * VSLY23 / 1e6
    lo, hi = q(dv)
    c["VLW, US$ million lower"], c["VLW, US$ million upper"] = lo / 1e6, hi / 1e6
    c["YLL VLW, US$ million"] = c["YLLs"] * VSLY23 / 1e6
    c["YLD VLW, US$ million"] = c["YLDs"] * VSLY23 / 1e6
    c["VLW per child, US$"] = c["DALYs"] * VSLY23 / pop23
    c["VLW per child, US$ lower"], c["VLW per child, US$ upper"] = lo / pop23, hi / pop23
    c["GDP, US$ million"] = GDPTOT[:, -1] / 1e6
    c["VLW/GDP, %"] = c["VLW, US$ million"] / c["GDP, US$ million"] * 100
    c["VLW/GDP, % lower"] = c["VLW, US$ million lower"] / c["GDP, US$ million"] * 100
    c["VLW/GDP, % upper"] = c["VLW, US$ million upper"] / c["GDP, US$ million"] * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        c["VLW per incident case, US$"] = c["DALYs"] * VSLY23 / c["Incidence"]
        c["VLW per death, US$"] = c["DALYs"] * VSLY23 / c["Deaths"]
        c["Deaths/incidence"] = c["Deaths"] / c["Incidence"]
    c["US$ per I$ (2023 price-level ratio)"] = XR
    c.loc[~MON, [x for x in c.columns if "VLW" in x or "VSLY" in x or "GDP" in x]] = np.nan
    return c


C23 = country_frame()
C23["Rank by VLW"] = C23["VLW, US$ million"].rank(ascending=False, method="min")


# Group results (2023 and trend)
def group_2023(mask, z=None):
    d23 = D23 if z is None else D23_COR
    out = {}
    mk = mask
    mm = mask & MON
    pop = POP[2, :, :, Y23]
    out["Countries"] = int(mk.sum())
    out["Countries valued"] = int(mm.sum())
    for m, lab in MEAS.items():
        val = N["val"][MI[m], 2, :, mk, Y23].sum()
        dd = d23[m][:, :, mk].sum(axis=(1, 2))
        out[lab] = (val, *q(dd))
        num_a = N["val"][MI[m], 2][:, mk, Y23].sum(axis=1)
        pop_a = pop[:, mk].sum(axis=1)
        asr = (W * num_a / pop_a).sum() * 1e5
        dasr = (W[None] * d23[m][:, :, mk].sum(axis=2) / pop_a[None]).sum(axis=1) * 1e5
        out[lab + " ASR"] = (asr, *q(dasr))
    out["Children <10"] = pop[:, mk].sum()
    vlw = (N["val"][MI[2], 2][:, mm, Y23].sum(axis=0) * VSLY23[mm]).sum()
    dv = (d23[2][:, :, mm].sum(axis=1) * VSLY23[mm][None]).sum(axis=1)
    out["VLW"] = (vlw, *q(dv))
    out["YLL VLW"] = (N["val"][MI[4], 2][:, mm, Y23].sum(axis=0) * VSLY23[mm]).sum()
    out["YLD VLW"] = (N["val"][MI[3], 2][:, mm, Y23].sum(axis=0) * VSLY23[mm]).sum()
    popm = pop[:, mm].sum()
    gdp = GDPTOT[mm, -1].sum()
    out["VLW per child"] = (vlw / popm, *(np.array(q(dv)) / popm))
    out["VLW/GDP %"] = (vlw / gdp * 100, *(np.array(q(dv)) / gdp * 100))
    out["DALY coverage %"] = N["val"][MI[2], 2][:, mm, Y23].sum() / N["val"][MI[2], 2][:, mk, Y23].sum() * 100
    out["Share <5 %"] = N["val"][MI[2], 2, 0, mk, Y23].sum() / N["val"][MI[2], 2][:, mk, Y23].sum() * 100
    out["VLW per incident case"] = vlw / N["val"][MI[6], 2][:, mm, Y23].sum()
    return out


G23 = {g: group_2023(GROUPS[g]) for g in GN}
G23C = {g: group_2023(GROUPS[g], z="cor") for g in GN}
require_agreement("Income-group VLW sums to All LMICs", "2023",
   sum(G23[g]["VLW"][0] for g in INC) / G23["All LMICs"]["VLW"][0] - 1, 1e-9)
require_agreement("Country VLW sums to All LMICs", "2023", C23["VLW, US$ million"].sum() * 1e6 / G23["All LMICs"]["VLW"][0] - 1, 1e-9)
require_agreement("Country DALY = YLL + YLD (VLW split)", "2023 All LMICs",
   (G23["All LMICs"]["YLL VLW"] + G23["All LMICs"]["YLD VLW"]) / G23["All LMICs"]["VLW"][0] - 1, .01)

# trend (both sexes), DALY draws -> VLW draws with current-year and fixed 2023 VSLY
TREND = []
for g in GN:
    mk = GROUPS[g]
    mm = mk & MONH
    for j, yr in enumerate(YEARS):
        pop_a = POP[2][:, mk, j].sum(axis=1)
        row = {"Group": g, "Year": yr}
        for m, lab in MEAS.items():
            num_a = N["val"][MI[m], 2][:, mk, j].sum(axis=1)
            row[lab] = num_a.sum()
            row[lab + " ASR"] = (W * num_a / pop_a).sum() * 1e5
        dd = D_DALY[:, :, mk, j]
        row["DALYs lower"], row["DALYs upper"] = q(dd.sum(axis=(1, 2)))
        lo, hi = q((W[None] * dd.sum(axis=2) / pop_a[None]).sum(axis=1) * 1e5)
        row["DALYs ASR lower"], row["DALYs ASR upper"] = lo, hi
        dv = (D_DALY[:, :, mm, j].sum(axis=1) * VSLY[mm, j][None]).sum(axis=1)
        dvf = (D_DALY[:, :, mm, j].sum(axis=1) * VSLY23[mm][None]).sum(axis=1)
        row["VLW"] = (N["val"][MI[2], 2][:, mm, j].sum(axis=0) * VSLY[mm, j]).sum()
        row["VLW lower"], row["VLW upper"] = q(dv)
        row["VLW fixed 2023 VSLY"] = (N["val"][MI[2], 2][:, mm, j].sum(axis=0) * VSLY23[mm]).sum()
        row["VLW fixed lower"], row["VLW fixed upper"] = q(dvf)
        row["Children <10"] = pop_a.sum()
        row["GDP"] = GDPTOT[mm, j].sum()
        TREND.append(row)
TREND = pd.DataFrame(TREND)


def pct_change(g, m, asr=False):
    mk = GROUPS[g]
    if m == 2:
        d = D_DALY
    else:
        d = draws(m)
    pop0, pop1 = POP[2][:, mk, 0].sum(axis=1), POP[2][:, mk, -1].sum(axis=1)
    if asr:
        a0 = (W[None] * d[:, :, mk, 0].sum(axis=2) / pop0[None]).sum(axis=1)
        a1 = (W[None] * d[:, :, mk, -1].sum(axis=2) / pop1[None]).sum(axis=1)
        p0 = (W * N["val"][MI[m], 2][:, mk, 0].sum(axis=1) / pop0).sum()
        p1 = (W * N["val"][MI[m], 2][:, mk, -1].sum(axis=1) / pop1).sum()
    else:
        a0, a1 = d[:, :, mk, 0].sum(axis=(1, 2)), d[:, :, mk, -1].sum(axis=(1, 2))
        p0, p1 = N["val"][MI[m], 2][:, mk, 0].sum(), N["val"][MI[m], 2][:, mk, -1].sum()
    return ((p1 / p0 - 1) * 100, *q((a1 / a0 - 1) * 100))


# Inequality
def conc_index(y, w, rankvar):
    """Weighted concentration index of y (rate) across units ranked by rankvar (ascending income)."""
    o = np.argsort(rankvar, kind="stable")
    y, w, rv = y[o], w[o] / w[o].sum(), np.asarray(rankvar)[o]
    r = np.cumsum(w) - w / 2
    for t in np.unique(rv[pd.Series(rv).duplicated(keep=False).values]):   # ties share the group's mid-rank
        k_ = rv == t
        r[k_] = (np.cumsum(w)[k_].max() + (np.cumsum(w)[k_] - w[k_]).min()) / 2
    mu = (w * y).sum()
    return 2 * (w * (y - mu) * (r - .5)).sum() / mu, r, y, w


def sii(y, w, r):
    X = np.column_stack([np.ones_like(r), r])
    Wm = np.diag(w)
    b = np.linalg.solve(X.T @ Wm @ X, X.T @ Wm @ y)
    return b[1]


INEQ = []
CURVES = {}
SCAT = {}
for yr in (1990, 2023):
    j = yr - 1990
    mk = MONH
    pop = POP[2][:, mk, j].sum(axis=0)
    rankv = G[mk, j]
    metrics = {
        "DALY rate per 100,000": (N["val"][MI[2], 2][:, mk, j].sum(axis=0) / pop * 1e5,
                                  D_DALY[:, :, mk, j].sum(axis=1) / pop[None] * 1e5),
        "Death rate per 100,000": (N["val"][MI[1], 2][:, mk, j].sum(axis=0) / pop * 1e5, None),
        "VLW per child, US$": (N["val"][MI[2], 2][:, mk, j].sum(axis=0) * VSLY[mk, j] / pop,
                              D_DALY[:, :, mk, j].sum(axis=1) * VSLY[mk, j][None] / pop[None]),
        "VLW/GDP, %": (N["val"][MI[2], 2][:, mk, j].sum(axis=0) * VSLY[mk, j] / GDPTOT[mk, j] * 100, None),
    }
    for lab, (y, yd) in metrics.items():
        ci, r, ys, ws = conc_index(y, pop, rankv)
        s = sii(ys, ws, r)
        row = {"Year": yr, "Indicator": lab, "Countries": int(mk.sum()), "Concentration index": ci, "SII": s}
        if yd is not None:
            cis = np.array([conc_index(yd[k], pop, rankv)[0] for k in range(ND)])
            siis = np.array([sii(yd[k][np.argsort(rankv, kind="stable")], ws, r) for k in range(ND)])
            row["CI lower"], row["CI upper"] = q(cis)
            row["SII lower"], row["SII upper"] = q(siis)
        INEQ.append(row)
        if lab in ("DALY rate per 100,000", "VLW per child, US$"):
            num = ys * ws
            CURVES[(yr, lab)] = (np.r_[0, np.cumsum(ws)], np.r_[0, np.cumsum(num) / num.sum()])
            o_ = np.argsort(rankv, kind="stable")
            ids = np.where(mk)[0][o_]
            SCAT[(yr, lab)] = pd.DataFrame({
                "Year": yr, "Indicator": lab, "location_id": [LIDS[i] for i in ids],
                "Country": [NAME[LIDS[i]] for i in ids], "Income group": INCG[ids],
                "GNI pc, I$": rankv[o_], "Relative income rank": r, "Value": ys,
                "Children <10": pop[o_], "Weight": ws,
                "Cumulative share of children": np.cumsum(ws), "Cumulative share of burden": np.cumsum(num) / num.sum(),
                "Fitted (SII line)": (lambda b_: b_[0] + b_[1] * r)(np.linalg.solve(
                    np.column_stack([np.ones_like(r), r]).T @ np.diag(ws) @ np.column_stack([np.ones_like(r), r]),
                    np.column_stack([np.ones_like(r), r]).T @ np.diag(ws) @ ys))})
INEQ = pd.DataFrame(INEQ)
for (y_, lab_), s_ in SCAT.items():
    fit = s_["Fitted (SII line)"].values
    sl = (fit[-1] - fit[0]) / (s_["Relative income rank"].values[-1] - s_["Relative income rank"].values[0])
    require_agreement(f"Fig. 3 regression slope = tabulated SII ({lab_}, {y_})", "weighted LS",
       sl - INEQ.query("Year==@y_ and Indicator==@lab_")["SII"].iloc[0], 1e-6)

# shares by income group (2023, monetary set)
SH = []
for g in INC:
    mk = GROUPS[g] & MON
    SH.append({"Income group": g,
               "Children <10": POP[2][:, mk, Y23].sum() / POP[2][:, MON, Y23].sum() * 100,
               "Deaths": N["val"][MI[1], 2][:, mk, Y23].sum() / N["val"][MI[1], 2][:, MON, Y23].sum() * 100,
               "DALYs": N["val"][MI[2], 2][:, mk, Y23].sum() / N["val"][MI[2], 2][:, MON, Y23].sum() * 100,
               "VLW (country VSLY)": C23.loc[mk, "VLW, US$ million"].sum() / C23.loc[MON, "VLW, US$ million"].sum() * 100,
               "GDP": GDPTOT[mk, -1].sum() / GDPTOT[MON, -1].sum() * 100})
SH = pd.DataFrame(SH)
SH["VLW share − DALY share, pp"] = SH["VLW (country VSLY)"] - SH["DALYs"]
# share UIs from the country draws (population and GDP are not modelled with uncertainty)
_sd = {"Deaths": D23[1].sum(axis=1)[:, MON], "DALYs": D23[2].sum(axis=1)[:, MON],
       "VLW (country VSLY)": (D23[2].sum(axis=1) * VSLY23[None])[:, MON]}
for col_, dr_ in _sd.items():
    tot_ = dr_.sum(axis=1)
    lo_hi = [q(dr_[:, GROUPS[g][MON]].sum(axis=1) / tot_ * 100) for g in INC]
    SH[col_ + " lower"], SH[col_ + " upper"] = [x[0] for x in lo_hi], [x[1] for x in lo_hi]
gap_ = [q((_sd["VLW (country VSLY)"][:, GROUPS[g][MON]].sum(axis=1) / _sd["VLW (country VSLY)"].sum(axis=1)
           - _sd["DALYs"][:, GROUPS[g][MON]].sum(axis=1) / _sd["DALYs"].sum(axis=1)) * 100) for g in INC]
SH["VLW share − DALY share, pp lower"], SH["VLW share − DALY share, pp upper"] = [x[0] for x in gap_], [x[1] for x in gap_]
UNIF_VSLY = (VSLY23[MON] * POP[2][:, MON, Y23].sum(axis=0)).sum() / POP[2][:, MON, Y23].sum()
uni = C23["DALYs"] * UNIF_VSLY
require_agreement("Uniform VSLY: VLW share = DALY share (identity)", "LIC",
   uni[GROUPS["Low income"] & MON].sum() / uni[MON].sum() * 100 - SH.set_index("Income group").loc["Low income", "DALYs"], 1e-9)
C23["VLW uniform VSLY, US$ million"] = np.where(MON, uni / 1e6, np.nan)
C23["Change with uniform VSLY, US$ million"] = C23["VLW uniform VSLY, US$ million"] - C23["VLW, US$ million"]

# Priority matrix
pm = C23[MON].copy()
xmed, ymed = pm["DALYs ASR"].median(), pm["VLW/GDP, %"].median()


def quadrant(r):
    hx, hy = r["DALYs ASR"] >= xmed, r["VLW/GDP, %"] >= ymed
    return {(True, True): "Q1 high health & high economic burden", (True, False): "Q2 high health, low economic",
            (False, True): "Q3 low health, high economic", (False, False): "Q4 low & low"}[(hx, hy)]


C23["Priority quadrant"] = np.nan
C23.loc[MON, "Priority quadrant"] = pm.apply(quadrant, axis=1)

# Sensitivity analyses (2023)
DALY23 = N["val"][MI[2], 2][:, :, Y23].sum(axis=0)
YLL23 = N["val"][MI[4], 2][:, :, Y23].sum(axis=0)
gus23, g23, e23 = GUS[-1], G[:, -1], EMED[:, -1]


def summ(label, vsly_vec, mask=MON, note="", per=DALY23, gdp=None):
    v = per * vsly_vec
    gdp = GDPTOT[:, -1] if gdp is None else gdp
    tot = np.nansum(v[mask])
    lic = np.nansum(v[mask & GROUPS["Low income"]]) / tot * 100
    lmc = np.nansum(v[mask & GROUPS["Lower middle income"]]) / tot * 100
    return {"Scenario": label, "Countries": int(mask.sum()), "VLW, US$ billion": tot / 1e9,
            "Change vs main, %": np.nan, "LIC share, %": lic, "LMC share, %": lmc,
            "VLW/GDP, %": tot / np.nansum(gdp[mask]) * 100, "Note": note}


SENS = [summ("Main: VSL elasticity 1.5, floor 20 × GNI, VSL/GNI(US) 160, undiscounted", VSLY23,
             note="Robinson et al. 2019 reference case; VSLY = VSL / e(median age)")]
for eps in (1.0, 1.2, 2.0):
    SENS.append(summ(f"Income elasticity {eps}", vsl(g23, gus23, eps=eps) / e23 * XR))
SENS.append(summ("No VSL floor (elasticity 1.5)", vsl(g23, gus23, floor=None) / e23 * XR,
                 note=f"floor binds in {int(FLOOR[MON, -1].sum())} countries in 2023"))
for rr_ in (100, 200):
    SENS.append(summ(f"US VSL/GNI ratio {rr_}", vsl(g23, gus23, ratio_us=rr_) / e23 * XR))
SENS.append(summ("VSLY annuitised at 3% over e(median age)", VSL[:, -1] / annuity(e23)))
SENS.append(summ("Uniform VSLY (child-population-weighted mean)", np.full(NC, UNIF_VSLY),
                 note=f"US${UNIF_VSLY:,.0f} per DALY"))
SENS.append(summ("Observed 2023 GNI only (exclude imputed income)", VSLY23, mask=MON & OBS23,
                 note=f"{int((MON & ~OBS23).sum())} countries with imputed 2023 GNI dropped"))
SENS.append(summ("Reported in 2023 I$ (PPP, no exchange-rate conversion)", VSLY23 / XR, gdp=GDPTOT[:, -1] / XR,
                 note="same VSL transfer; amounts left in international dollars"))
k = pd.read_csv(EXT / "k_threshold_130.csv").set_index("location_id").loc[LIDS]
SENS.append(summ("Health opportunity cost: country threshold per DALY × DALY", k.k_main.values * GDPPC[:, -1],
                 note="k_main × GDP per capita (Pichon-Riviere 2023 / Ochalek 2018); budget perspective"))
SENS.append(summ("1 × GDP per capita per DALY", GDPPC[:, -1], note="WHO-CHOICE-type benchmark"))
SENS.append(summ("Premature mortality only (YLL × VSLY)", VSLY23, per=YLL23))
SENS = pd.DataFrame(SENS)
SENS["Change vs main, %"] = (SENS["VLW, US$ billion"] / SENS.loc[0, "VLW, US$ billion"] - 1) * 100
corr = G23C["All LMICs"]["VLW"]
ind = G23["All LMICs"]["VLW"]
UI_TAB = pd.DataFrame([{"Group": g, "Assumption": a, "VLW, US$ million": v[0] / 1e6, "Lower": v[1] / 1e6,
                        "Upper": v[2] / 1e6, "DALYs": d[0], "DALYs lower": d[1], "DALYs upper": d[2]}
                       for g in GN for a, src in (("Independent between countries (main)", G23),
                                                  ("Fully correlated between countries", G23C))
                       for v, d in [(src[g]["VLW"], src[g]["DALYs"])]])
FLOOR_N = int(FLOOR[MON, -1].sum())
FLOOR_DALY = DALY23[MON & FLOOR[:, -1]].sum() / DALY23[MON].sum() * 100
FLOOR_NAMES = ", ".join(NAME[LIDS[i]] for i in np.where(MON & FLOOR[:, -1])[0])

# External comparisons
# (a) IICC-3 population-based registries vs GBD incidence (<5, 5–9), LMIC registries
reg = pd.read_csv(DATA / "05-retinoblastoma.csv", encoding="utf-8-sig")
alias = {"VIET NAM": "Viet Nam", "ISLAMIC REPUBLIC OF IRAN": "Iran (Islamic Republic of)", "TURKEY": "Türkiye",
         "BOLIVIA": "Bolivia (Plurinational State of)", "RUSSIAN FEDERATION": "Russian Federation",
         "LIBYA": "Libya", "EGYPT": "Egypt", "BELARUS": "Belarus"}
up = {n.upper(): l for n, l in zip(cw.location_name, cw.location_id)}
reg["location_id"] = reg.country_or_territory.map(lambda s: up.get(s, up.get(alias.get(s, s).upper())))
reg = reg[reg.location_id.notna()].copy()
reg["location_id"] = reg.location_id.astype(int)
reg["mid_year"] = reg.observation_period.map(lambda s: int(np.floor(np.mean([int(x) for x in str(s).split("-")]) + .5)))
reg["national"] = ~reg.registry_population_label.str.contains(",") | reg.registry_population_label.str.contains("registries")
rows = []
for _, r in reg.iterrows():
    i, j = LIDX[r.location_id], int(np.clip(r.mid_year, 1990, 2023)) - 1990
    for a, col in ((0, "rate_0_4_per_million"), (1, "rate_5_9_per_million")):
        g_rate = N["val"][MI[6], 2, a, i, j] / POP[2, a, i, j] * 1e6
        g_lo = N["lower"][MI[6], 2, a, i, j] / POP[2, a, i, j] * 1e6
        g_hi = N["upper"][MI[6], 2, a, i, j] / POP[2, a, i, j] * 1e6
        rows.append({"Country": NAME[r.location_id], "Income group": INCG[i], "Registry": r.registry_population_label,
                     "Period": r.observation_period, "Matched GBD year": j + 1990, "Age": ["<5", "5–9"][a],
                     "Registry cases 0–14": r.cases_0_14, "Registry rate per million": r[col],
                     "GBD rate per million": g_rate, "GBD lower": g_lo, "GBD upper": g_hi})
REGCMP = pd.DataFrame(rows)
REGCMP["Ratio GBD/registry"] = REGCMP["GBD rate per million"] / REGCMP["Registry rate per million"].replace(0, np.nan)
REGCMP["Registry within GBD UI"] = REGCMP["Registry rate per million"].between(REGCMP["GBD lower"], REGCMP["GBD upper"])
rc = REGCMP[REGCMP["Registry rate per million"] > 0]
REG_RHO = spearmanr(rc["Registry rate per million"], rc["GBD rate per million"]).correlation
REG_RATIO = rc.groupby("Age")["Ratio GBD/registry"].median()
REG_WITHIN = rc["Registry within GBD UI"].mean() * 100

# (b) published GBD 2023 figures and the GBD global/regional aggregates (R2)
r2 = read_batch(DATA, "R2", list(REGION_LID.values()))
PUB = []
if len(r2):
    r2 = r2[(r2.year == 2023) & (r2.metric == 1) & (r2.sex == 3)]
    for reg_name, lid in REGION_LID.items():
        d = r2[r2.location == lid].groupby("measure")[["val", "lower", "upper"]].sum()
        mk = GROUPS.get(reg_name, np.zeros(NC, bool))
        for m, lab in MEAS.items():
            if m not in d.index:
                continue
            own = N["val"][MI[m], 2][:, mk, Y23].sum() if mk.any() else np.nan
            PUB.append({"Location": reg_name, "Measure": lab, "GBD aggregate <10, 2023": d.loc[m, "val"],
                        "GBD lower (sum of <5 and 5–9 bounds, approx.)": d.loc[m, "lower"],
                        "GBD upper (sum of <5 and 5–9 bounds, approx.)": d.loc[m, "upper"],
                        "Sum of cohort LMICs (this study)": own if reg_name != "Global" else G23["All LMICs"][lab][0]})
PUB = pd.DataFrame(PUB)
if len(PUB):
    PUB["Cohort / aggregate, %"] = PUB["Sum of cohort LMICs (this study)"] / PUB["GBD aggregate <10, 2023"] * 100
LANCET = {"Deaths": (3250, 1280, 7100), "DALYs": (285000, 113000, 622000), "Incidence": (5400, 3020, 9580)}
LANCET_TAB = pd.DataFrame([{"Measure": m, "GBD 2023 Cancer Collaborators 2025, global all ages": v[0],
                            "Lower": v[1], "Upper": v[2],
                            "GBD Results Tool, global <10 (R2)": (PUB.query("Location=='Global' and Measure==@m")["GBD aggregate <10, 2023"].sum() if len(PUB) else np.nan),
                            "This study, 130 LMICs <10": G23["All LMICs"][m][0]} for m, v in LANCET.items()])
LANCET_TAB["Global <10 / published, %"] = LANCET_TAB["GBD Results Tool, global <10 (R2)"] / LANCET_TAB["GBD 2023 Cancer Collaborators 2025, global all ages"] * 100
LANCET_TAB["LMIC <10 / published, %"] = LANCET_TAB["This study, 130 LMICs <10"] / LANCET_TAB["GBD 2023 Cancer Collaborators 2025, global all ages"] * 100

# (c) previous estimate (ages 0–19 y, VSL/(HALE/2)) — bridge table
oldt = pd.read_csv(DATA / "Table_10_Retinoblastoma.csv", skiprows=2)
br = C23[MON][["location_id", "Country", "Income group", "DALYs", "VSLY, US$", "VLW, US$ million"]].merge(
    oldt[["location_id", "DALY", "VSL_IntUSD", "HALE_years", "VLW_nondisc_millionIntUSD"]], on="location_id")
i_br = br.location_id.map(LIDX).values
br["Previous estimate, US$ million"] = br.VLW_nondisc_millionIntUSD * XR[i_br]
br["Previous VSL / e(median age)"] = br.DALY * br.VSL_IntUSD * XR[i_br] / e23[i_br] / 1e6
br["VSL new, US$"] = VSL[i_br, -1]
BRIDGE = pd.DataFrame([{
    "Step": s, "Countries": len(br), "VLW, US$ million": v, "Change from previous step, US$ million": np.nan} for s, v in [
    ("Previous estimate in I$ (0–19 y; VSL/(HALE/2); income elasticity 1.0; undiscounted)", br.VLW_nondisc_millionIntUSD.sum()),
    ("Convert to 2023 US$ with each country's price-level ratio", br["Previous estimate, US$ million"].sum()),
    ("Replace HALE/2 by e(median age)", br["Previous VSL / e(median age)"].sum()),
    ("Replace previous VSL by this study's VSL (GNI-based, elasticity 1.5)", br["VLW, US$ million"].sum()),
]])
BRIDGE["Change from previous step, US$ million"] = BRIDGE["VLW, US$ million"].diff()
require_agreement("Same DALY numerator as previous estimate (max abs. diff; old table rounded to 0.01)", f"{len(br)} countries",
   float((br.DALYs - br.DALY).abs().max()), 0.01)

# (d) published cost studies (lit_rb_cost.csv)
LIT_FILE = EXT / "lit_rb_cost.csv"
lit = pd.read_csv(LIT_FILE, encoding="utf-8-sig")
LIT = []
for _, r in lit.iterrows():
    if r.iso3 not in ISO:
        continue
    i = ISO.index(r.iso3)
    j = int(np.clip(r.data_year, 1990, 2023)) - 1990
    pyr = int(np.clip(r.price_year, 1990, 2023)) - 1990
    val_ppp23 = r.value * DEFL[pyr] if r.currency == "USD" else np.nan      # 2023 US$ (US GDP deflator)
    model = N["val"][MI[2], 2][:, i, j].sum() * VSLY[i, j] / N["val"][MI[6], 2][:, i, j].sum()
    LIT.append({"Study": r.study, "Country": NAME[LIDS[i]], "Income group": INCG[i], "Cost type": r.cost_type,
                "Perspective": ("Indirect" if r.cost_type.startswith("Indirect") else
                                "Provider" if r.cost_type.startswith("Provider") else "Household direct"),
                "n": r.n, "Data year": r.data_year,
                "Reported value": r.value, "Currency/price year": f"{r.currency} {r.price_year}",
                "Literature cost per case, 2023 US$": val_ppp23, "Model VLW per incident case, 2023 US$": model,
                "Literature / model, %": val_ppp23 / model * 100, "Verification": r.verified})
LIT = pd.DataFrame(LIT)

# (e) mortality-to-incidence ratio vs Global Retinoblastoma Study Group 3-year survival (LIC 57.3%)
MIR = []
for g in GMAIN:
    mk = GROUPS[g]
    MIR.append({"Group": g, "Deaths": N["val"][MI[1], 2][:, mk, Y23].sum(), "Incidence": N["val"][MI[6], 2][:, mk, Y23].sum()})
MIR = pd.DataFrame(MIR)
MIR["Deaths/incidence"] = MIR.Deaths / MIR.Incidence
MIR["1 − deaths/incidence"] = 1 - MIR["Deaths/incidence"]
MIR["GRBSG 2022 3-year survival (LIC only)"] = np.where(MIR.Group == "Low income", .573, np.nan)

# Figures
sdata = {}


def add_sd(fig, panel_, df):
    sdata.setdefault(fig, {})[panel_] = df


# ---- Fig 1: trends and composition
fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.2))
t = TREND[TREND.Group == "All LMICs"]
ax = axs[0]
ax.fill_between(YEARS, t["VLW lower"] / 1e9, t["VLW upper"] / 1e9, color="#197068", alpha=.18, lw=0)
ax.plot(YEARS, t["VLW"] / 1e9, color="#197068", lw=2.4, label="Valued at each year's VSLY")
ax.fill_between(YEARS, t["VLW fixed lower"] / 1e9, t["VLW fixed upper"] / 1e9, color="#7F7F7F", alpha=.15, lw=0)
ax.plot(YEARS, t["VLW fixed 2023 VSLY"] / 1e9, color="#7F7F7F", ls="--", label="Valued at 2023 VSLY")
ax.set_ylabel("VLW, billion US$ (2023 prices)")
ax.legend(frameon=False, loc="upper left")
ax.set_title(f"Annual welfare loss, {int(MONH.sum())} countries", loc="left")
add_sd("Fig1", "a", t[["Year", "VLW", "VLW lower", "VLW upper", "VLW fixed 2023 VSLY", "VLW fixed lower", "VLW fixed upper"]])
ax = axs[1]
rows = []
for g in GMAIN:
    tt = TREND[TREND.Group == g]
    ax.fill_between(YEARS, tt["DALYs ASR lower"], tt["DALYs ASR upper"], color=INC_COL[g], alpha=.12, lw=0)
    ax.plot(YEARS, tt["DALYs ASR"], color=INC_COL[g], ls="--" if g == "All LMICs" else "-", label=g)
    rows.append(tt[["Group", "Year", "DALYs ASR", "DALYs ASR lower", "DALYs ASR upper"]])
ax.set_ylabel("Age-standardised DALY rate, <10 y (per 100,000)")
ax.set_title("Health loss by income group", loc="left")
ax.legend(frameon=False, loc="upper right")
add_sd("Fig1", "b", pd.concat(rows))
ax = axs[2]
cats = GMAIN
x = np.arange(len(cats))
yll = np.array([G23[g]["YLLs"][0] for g in cats])
yld = np.array([G23[g]["YLDs"][0] for g in cats])
u5 = np.array([N["val"][MI[2], 2, 0][GROUPS[g], Y23].sum() for g in cats])
o5 = np.array([N["val"][MI[2], 2, 1][GROUPS[g], Y23].sum() for g in cats])
ax.bar(x - .2, yll / 1e3, .38, color="#264653", label="YLLs")
ax.bar(x - .2, yld / 1e3, .38, bottom=yll / 1e3, color="#E9C46A", label="YLDs")
ax.bar(x + .2, u5 / 1e3, .38, color="#2A9D8F", label="Age <5")
ax.bar(x + .2, o5 / 1e3, .38, bottom=u5 / 1e3, color="#A8DADC", label="Age 5–9")
dlo = np.array([G23[g]["DALYs"][1] for g in cats])
dhi = np.array([G23[g]["DALYs"][2] for g in cats])
dtot = yll + yld
for off in (-.2, .2):
    ax.errorbar(x + off, dtot / 1e3, yerr=[(dtot - dlo) / 1e3, (dhi - dtot) / 1e3], fmt="none", ecolor="k",
                elinewidth=.7, capsize=2.2, capthick=.7)
for xi, a, b, h in zip(x, yll, yld, dhi):
    ax.text(xi - .2, h / 1e3 * 1.01, f"{a / (a + b) * 100:.1f}%\nYLL", ha="center", va="bottom", fontsize=6.3)
ax.set_xticks(x)
ax.set_xticklabels(["All LMICs", "Low", "Lower-middle", "Upper-middle"], fontsize=8)
ax.set_ylabel("DALYs, 2023 (thousands)")
ax.set_ylim(0, dhi.max() / 1e3 * 1.14)
ax.set_title("Composition of DALYs, 2023", loc="left")
ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper right")
add_sd("Fig1", "c", pd.DataFrame({"Group": cats, "YLLs": yll, "YLDs": yld, "DALYs <5": u5, "DALYs 5–9": o5,
                                    "DALYs total": dtot, "DALYs lower": dlo, "DALYs upper": dhi}))
for k_, ax in enumerate(axs):
    if k_ < 2:
        ax.set(xlabel="Year", ylim=(0, None), xticks=[1990, 2000, 2010, 2023])
    panel(ax, k_)
save(fig, "Fig1_burden_trends_composition_1990_2023")

# ---- Fig 2: maps 2023
world = gpd.read_file(DATA / "df_world2.geojson").to_crs(4326)
world = world[world.brk_a3 != "ATA"]
world["lid"] = world.location_id.fillna(-1).astype(int)
mp = world.merge(C23.rename(columns={"location_id": "lid"}), on="lid", how="left")
require_agreement("Cohort countries drawn on map", "130 (micro-states absent from map file)",
   int(130 - mp.lid.isin(LIDS).groupby(mp.lid).any().sum()), 15)
fig = plt.figure(figsize=(14, 8.6))
specs = [("DALYs ASR", "Age-standardised DALY rate, <10 y (per 100,000)", [2, 5, 10, 20, 40]),
         ("VLW, US$ million", "Value of lost welfare (million 2023 US$)", [.1, .5, 2.5, 10, 50]),
         ("VLW per child, US$", "VLW per child <10 y (2023 US$)", [.1, .25, .5, 1, 2]),
         ("VLW/GDP, %", "VLW as share of GDP (%)", [.001, .0025, .005, .01, .02])]
colors = ["#FEF0D9", "#FDD49E", "#FDBB84", "#FC8D59", "#E34A33", "#B30000"]
for k_, (col, title, cuts) in enumerate(specs):
    ax = fig.add_subplot(2, 2, k_ + 1)
    world.plot(ax=ax, color="#EEEEEE", edgecolor="white", linewidth=.22)
    d = mp[mp[col].notna()].copy()
    classes = np.searchsorted(cuts, d[col], side="right")
    for i, color in enumerate(colors):
        if (classes == i).any():
            d.loc[classes == i].plot(ax=ax, color=color, edgecolor="white", linewidth=.24)
    miss = mp[mp.iso3.notna() & mp[col].isna()]
    if len(miss):
        miss.plot(ax=ax, color="white", edgecolor=".55", hatch="////", linewidth=.3)
    labels = [f"<{cuts[0]:g}"] + [f"{a:g}–<{b:g}" for a, b in zip(cuts, cuts[1:])] + [f"≥{cuts[-1]:g}"]
    handles = [Patch(facecolor=c, label=v) for c, v in zip(colors, labels)]
    handles += [Patch(fc="#EEEEEE", label="Not in LMIC cohort")]
    if len(miss):
        handles += [Patch(fc="white", ec=".55", hatch="////", label="No income data")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(.0, .02), frameon=False, ncol=1, fontsize=6.8,
              handlelength=1.3, labelspacing=.15)
    ax.set(xlim=(-180, 180), ylim=(-58, 78))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", x=.055)
    panel(ax, k_, x=0)
    add_sd("Fig2", "abcd"[k_], C23[["location_id", "iso3", "Country", "Income group", col] +
                                   [c for c in (col + " lower", col + " upper") if c in C23]])
save(fig, "Fig2_maps_health_and_welfare_loss_2023")

# ---- Fig 3: inequality
fig, axs = plt.subplots(1, 3, figsize=(17, 4.9), gridspec_kw={"width_ratios": [1.2, 1, 1]})
ax = axs[0]
cats = ["Children <10", "Deaths", "DALYs", "VLW (country VSLY)", "GDP"]
xb = np.arange(len(cats))
for i, g in enumerate(INC):
    vals = SH.set_index("Income group").loc[g, cats].values.astype(float)
    ax.bar(xb + (i - 1) * .26, vals, width=.26, color=INC_COL[g], label=g)
    shr = SH.set_index("Income group").loc[g]
    lo_ = np.array([shr.get(c_ + " lower", v_) for c_, v_ in zip(cats, vals)], float)
    hi_ = np.array([shr.get(c_ + " upper", v_) for c_, v_ in zip(cats, vals)], float)
    has = np.array([c_ + " lower" in shr.index for c_ in cats])
    ax.errorbar((xb + (i - 1) * .26)[has], vals[has], yerr=[vals[has] - lo_[has], hi_[has] - vals[has]], fmt="none",
                ecolor="k", elinewidth=.7, capsize=1.8, capthick=.7)
    for xx, vv, hh in zip(xb + (i - 1) * .26, vals, hi_):
        ax.text(xx, max(vv, hh) + .8, f"{vv:.0f}", ha="center", fontsize=6.5)
ax.set_xticks(xb)
ax.set_xticklabels(["Children\n<10 y", "Deaths", "DALYs", "VLW\n(country VSLY)", "GDP"], fontsize=8)
ax.set_ylabel(f"Share of {int(MON.sum())}-country total, 2023 (%)")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Health burden vs welfare-loss shares", loc="left")
add_sd("Fig3", "a", SH)
# (b) slope index of inequality: country scatter of DALY rate vs relative income rank, 1990 and 2023
ax = axs[1]
YRC = {1990: "#8C8C8C", 2023: "#C44E52"}
DR = "DALY rate per 100,000"
pmax = max(SCAT[(y_, DR)]["Children <10"].max() for y_ in (1990, 2023))
YCAP = 200
N_OFF = {}
for y_ in (1990, 2023):
    s_ = SCAT[(y_, DR)]
    row_ = INEQ.query("Year==@y_ and Indicator==@DR").iloc[0]
    on = s_["Value"] <= YCAP
    ax.scatter(s_.loc[on, "Relative income rank"], s_.loc[on, "Value"], s=8 + 260 * s_.loc[on, "Children <10"] / pmax,
               color=YRC[y_], alpha=.55 if y_ == 2023 else .35, edgecolor="white", lw=.4, zorder=2 if y_ == 2023 else 1)
    ax.scatter(s_.loc[~on, "Relative income rank"], np.full((~on).sum(), YCAP * .985), marker="^", s=22,
               color=YRC[y_], alpha=.8, zorder=2, clip_on=False)
    N_OFF[y_] = int((~on).sum())
    ax.plot(s_["Relative income rank"], s_["Fitted (SII line)"], color=YRC[y_], lw=1.8, zorder=3,
            label=f"{y_}: SII {row_['SII']:.1f} ({row_['SII lower']:.1f} to {row_['SII upper']:.1f})".replace("-", "−"))
s23 = SCAT[(2023, DR)]
for _, r_ in s23.nlargest(6, "Children <10").iterrows():
    ax.annotate(r_.Country, (r_["Relative income rank"], r_["Value"]), xytext=(4, 4), textcoords="offset points",
                fontsize=6.5, color=".25")
ax.set(xlabel="Relative rank by GNI per capita (0 = poorest, 1 = richest)",
       ylabel="DALY rate, <10 y (per 100,000)", xlim=(0, 1), ylim=(0, YCAP))
ax.legend(frameon=False, loc="upper right", fontsize=7, title="Weighted regression (95% UI)", title_fontsize=7)
ax.text(.99, .80, f"Bubble area proportional to children <10\n▲ off scale (>{YCAP}): 1990 n = {N_OFF[1990]}, 2023 n = {N_OFF[2023]}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=".4")
ax.set_title("Slope index of inequality, DALY rate", loc="left")
add_sd("Fig3", "b", pd.concat([SCAT[(y_, DR)] for y_ in (1990, 2023)]))
# (c) concentration curves, each country a point on the curve
ax = axs[2]
ax.plot([0, 1], [0, 1], color=".5", lw=.8, ls=":")
sty = {(1990, "DALY rate per 100,000"): ("#C44E52", "--"), (2023, "DALY rate per 100,000"): ("#C44E52", "-"),
       (1990, "VLW per child, US$"): ("#264653", "--"), (2023, "VLW per child, US$"): ("#264653", "-")}
for key, (cx, cy) in CURVES.items():
    ci = INEQ.query("Year==@key[0] and Indicator==@key[1]")["Concentration index"].iloc[0]
    lab = ("DALYs" if key[1].startswith("DALY") else "VLW") + f" {key[0]} (CI {ci:+.2f})"
    ax.plot(cx, cy, color=sty[key][0], ls=sty[key][1], lw=1.3, label=lab.replace("-", "−"))
    s_ = SCAT[key]
    ax.scatter(s_["Cumulative share of children"], s_["Cumulative share of burden"], s=6 + 90 * s_["Weight"] / s_["Weight"].max(),
               facecolor=sty[key][0] if key[0] == 2023 else "white", edgecolor=sty[key][0], lw=.6, alpha=.8, zorder=3)
ax.set(xlabel="Cumulative share of children <10, ranked by GNI per capita (poorest first)",
       ylabel="Cumulative share of burden", xlim=(0, 1), ylim=(0, 1))
ax.legend(frameon=False, loc="lower right", fontsize=7)
ax.set_title("Concentration curves (points = countries)", loc="left")
add_sd("Fig3", "c", pd.concat(SCAT.values()))
for k_, ax in enumerate(axs):
    panel(ax, k_)
save(fig, "Fig3_inequality_shares_concentration")

# ---- Fig 4: priority matrix (a) and deaths-to-incidence ratio (b)
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axs[0]
d = C23[MON].copy()
sz = 12 + 600 * np.sqrt(d["VLW, US$ million"] / d["VLW, US$ million"].max())
for g in INC:
    dd = d[d["Income group"] == g]
    ax.scatter(dd["DALYs ASR"], dd["VLW/GDP, %"], s=sz[dd.index], color=INC_COL[g], alpha=.6, edgecolor="white",
               lw=.5, label=g)
ax.axvline(xmed, color=".4", lw=.8, ls="--")
ax.axhline(ymed, color=".4", lw=.8, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
lab_set = set(d.nlargest(10, "VLW, US$ million").Country) | set(d.nlargest(5, "VLW/GDP, %").Country)
lab = d[d.Country.isin(lab_set)].copy()
lab["lx"], lab["ly"] = np.log10(lab["DALYs ASR"]), np.log10(lab["VLW/GDP, %"])
lab["name"] = lab.Country.str.replace(" (Plurinational State of)", "", regex=False)
# simple label repulsion in log space (labels placed to the left of the bubble)
tx = lab.lx.values - .06
ty = lab.ly.values.copy()
w = np.array([len(n) * .033 for n in lab.name])
for _ in range(400):
    moved = False
    for a_ in range(len(ty)):
        for b_ in range(a_ + 1, len(ty)):
            if abs(tx[a_] - tx[b_]) < (w[a_] + w[b_]) / 2 and abs(ty[a_] - ty[b_]) < .075:
                sgn = 1 if ty[a_] >= ty[b_] else -1
                ty[a_] += .02 * sgn
                ty[b_] -= .02 * sgn
                moved = True
    if not moved:
        break
for (_, r), x_, y_ in zip(lab.iterrows(), tx, ty):
    ax.annotate(r["name"], (r["DALYs ASR"], r["VLW/GDP, %"]), xytext=(10 ** x_, 10 ** y_), fontsize=6.8,
                ha="right", va="center", arrowprops=dict(arrowstyle="-", lw=.4, color=".45"))
ax.text(xmed * 1.08, ax.get_ylim()[1] * .9, "Q1 high health &\nhigh economic burden", ha="left", va="top", fontsize=7.5, color=".3")
ax.text(.99, .01, "Q2 high health,\nlow economic", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=".3")
ax.text(.01, .99, "Q3 low health,\nhigh economic", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=".3")
ax.set(xlabel="Age-standardised DALY rate, <10 y (per 100,000), 2023", ylabel="VLW as share of GDP (%), 2023")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=7, color=INC_COL[g], alpha=.7, label=g) for g in INC] +
          [plt.Line2D([], [], ls="", label="Bubble area proportional to total VLW")], frameon=False, loc="lower right", bbox_to_anchor=(1, .08))
ax.set_title("Priority matrix: health vs economic burden", loc="left")
add_sd("Fig4", "a", d[["location_id", "Country", "Income group", "DALYs ASR", "VLW/GDP, %", "VLW, US$ million", "Priority quadrant"]])
# (b) GBD deaths / incidence vs national income
ax = axs[1]
d = C23[MON & (C23.Incidence > 1)].copy()
isq1 = d["Priority quadrant"].fillna("").str.startswith("Q1")
for g in INC:
    dd = d[d["Income group"] == g]
    q1_ = isq1[dd.index]
    ax.scatter(dd["GNI pc, I$"], dd["Deaths/incidence"], s=22, color=INC_COL[g], alpha=.8, label=g,
               edgecolor=np.where(q1_, "k", "white"), lw=np.where(q1_, .8, .3))
for g in INC:
    ax.axhline(MIR.set_index("Group").loc[g, "Deaths/incidence"], xmin=0, xmax=1, color=INC_COL[g], lw=.8, ls=":")
ax.axhline(1 - .573, color="#C44E52", ls="--", lw=1)
ax.text(d["GNI pc, I$"].min(), 1 - .573 + .015, "1 − 3-year survival, LICs (GRBSG 2022)", fontsize=6.8,
        color="#C44E52", ha="left", va="bottom")
for _, r_ in d.nlargest(4, "Deaths").iterrows():
    ax.annotate(r_.Country, (r_["GNI pc, I$"], r_["Deaths/incidence"]), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
ax.set(xscale="log", ylim=(0, 1.1), xlabel="GNI per capita, 2023 I$ (PPP)", ylabel="GBD deaths / incident cases, <10 y, 2023")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=6, color=INC_COL[g], label=g) for g in INC] +
          [plt.Line2D([], [], marker="o", ls="", ms=6, mfc="w", mec="k", label="Q1 priority country"),
           plt.Line2D([], [], color=".4", ls=":", lw=.8, label="Income-group ratio (pooled)")],
          frameon=False, fontsize=7, loc="upper right")
ax.set_title("Deaths-to-incidence ratio by national income", loc="left")
add_sd("Fig4", "b", d[["Country", "Income group", "GNI pc, I$", "Deaths", "Incidence", "Deaths/incidence", "Priority quadrant"]])
for k_, ax in enumerate(axs):
    panel(ax, k_)
save(fig, "Fig4_priority_matrix_mortality_incidence_2023")

# ---- Supplementary Fig 5: external comparisons (sensitivity analysis)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
ax = axs[0]
for a, mk_ in (("<5", "o"), ("5–9", "s")):
    dd = rc[rc.Age == a]
    for g in INC:
        e_ = dd[dd["Income group"] == g]
        ax.errorbar(e_["Registry rate per million"], e_["GBD rate per million"],
                    yerr=[e_["GBD rate per million"] - e_["GBD lower"], e_["GBD upper"] - e_["GBD rate per million"]],
                    fmt=mk_, ms=4, color=INC_COL[g], alpha=.7, elinewidth=.5, capsize=0,
                    label=f"{INC_SHORT[g]}, age {a}")
lim = [.25, 150]
ax.plot(lim, lim, color=".4", lw=.8)
ax.plot(lim, [l * .5 for l in lim], color=".6", lw=.6, ls=":")
ax.plot(lim, [l * 2 for l in lim], color=".6", lw=.6, ls=":")
ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Registry incidence (IICC-3), per million",
       ylabel="GBD 2023 incidence, per million")
ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper left")
ax.set_title(f"Incidence: GBD vs cancer registries (ρ = {REG_RHO:.2f})", loc="left")
add_sd("SuppFig5", "a", REGCMP)
ax = axs[1]
ax.scatter(br["Previous estimate, US$ million"], br["VLW, US$ million"], s=14, c=[INC_COL[g] for g in br["Income group"]], alpha=.7)
lim2 = [max(1e-3, br[["Previous estimate, US$ million", "VLW, US$ million"]].min().min() * .7),
        br[["Previous estimate, US$ million", "VLW, US$ million"]].max().max() * 1.4]
ax.plot(lim2, lim2, color=".4", lw=.8)
ax.set(xscale="log", yscale="log", xlim=lim2, ylim=lim2, xlabel="Previous estimate (converted to US$), million",
       ylabel="This study, million US$")
ax.set_title(f"Country VLW vs previous estimate (n = {len(br)})", loc="left")
add_sd("SuppFig5", "b", br)
ax = axs[2]
if len(LIT):
    MK = {"Household direct": "o", "Indirect": "s", "Provider": "^"}
    for per, mk_ in MK.items():
        e_ = LIT[LIT.Perspective == per]
        ax.scatter(e_["Model VLW per incident case, 2023 US$"], e_["Literature cost per case, 2023 US$"], marker=mk_,
                   s=22 + 3 * np.sqrt(e_["n"].fillna(10)), c=[INC_COL[g] for g in e_["Income group"]],
                   edgecolor="k", lw=.5, zorder=3, label=per)
    yo = np.log10(LIT["Literature cost per case, 2023 US$"].values)
    for k_, (_, r) in enumerate(LIT.iterrows()):
        au = r.Study.split(" et al.")[0] + " " + r.Study.split(", ")[0][-4:] if "et al." in r.Study else r.Study[:12]
        near = [j for j in range(len(LIT)) if j != k_ and abs(yo[j] - yo[k_]) < .25 and
                abs(np.log10(LIT["Model VLW per incident case, 2023 US$"].values[j] / r["Model VLW per incident case, 2023 US$"])) < .3]
        left = bool(near) and k_ % 2 == 1          # alternate sides when points crowd
        ax.annotate(f"{r.Country}, {au} ({r.Perspective.lower()})",
                    (r["Model VLW per incident case, 2023 US$"], r["Literature cost per case, 2023 US$"]), fontsize=6.2,
                    xytext=(-7 if left else 7, 0), textcoords="offset points", va="center", ha="right" if left else "left")
    lo_, hi_ = 3e3, 1e7
    for f, ls in ((1, "-"), (.1, "--"), (.01, ":")):
        ax.plot([lo_, hi_], [lo_ * f, hi_ * f], color=".5", lw=.8, ls=ls)
        ax.text(hi_, hi_ * f, f" y = {f:g}x", fontsize=6.5, va="center", color=".4")
    ax.set(xscale="log", yscale="log", xlim=(lo_, hi_ * 3), ylim=(3e2, 1e7),
           xlabel="Model VLW per incident case, 2023 US$", ylabel="Published cost per patient, 2023 US$")
    ax.legend(handles=[plt.Line2D([], [], marker=m_, ls="", mfc="w", mec="k", label=k_) for k_, m_ in MK.items()] +
              [Patch(color=INC_COL[g], label=INC_SHORT[g]) for g in INC], frameon=False, fontsize=6.5, loc="upper left", ncol=2)
ax.set_title(f"Published cost studies vs model ({LIT.Study.nunique()} studies, {len(LIT)} estimates)", loc="left")
add_sd("SuppFig5", "c", LIT)
for k_, ax in enumerate(axs):
    panel(ax, k_)
save(fig, "SuppFig5_external_comparisons")

# ---- Supplementary figures
# S1 tornado
fig, ax = plt.subplots(figsize=(8.5, 5))
s = SENS.iloc[1:].copy().sort_values("Change vs main, %")
ax.barh(np.arange(len(s)), s["Change vs main, %"], color=np.where(s["Change vs main, %"] > 0, "#E76F51", "#2A9D8F"))
ax.set_yticks(np.arange(len(s)))
ax.set_yticklabels(s.Scenario, fontsize=7.5)
ax.axvline(0, color="k", lw=.8)
for yy, (vv, tot) in enumerate(zip(s["Change vs main, %"], s["VLW, US$ billion"])):
    ax.text(vv, yy, f" {tot:.2f}", va="center", ha="left" if vv >= 0 else "right", fontsize=6.8)
ax.set_xlabel(f"Change in 2023 total VLW vs main estimate (US${SENS.loc[0, 'VLW, US$ billion']:.2f} billion), %")
add_sd("SuppFig1", "a", SENS)
save(fig, "SuppFig1_sensitivity_tornado")

# S2 simulation QC
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, (a, b, lab) in zip(axs, (("GBD lower", "Simulated 2.5%", "Lower bound"), ("GBD upper", "Simulated 97.5%", "Upper bound"))):
    dd = SIMQC[SIMQC[a] > 0]
    ax.scatter(dd[a], dd[b], s=6, alpha=.5, color="#264653")
    l_ = [dd[[a, b]].min().min(), dd[[a, b]].max().max()]
    ax.plot(l_, l_, color="#E76F51", lw=.8)
    ax.set(xscale="log", yscale="log", xlabel=f"{a} (DALY, 2023)", ylabel=b)
    ax.set_title(lab, loc="left")
for k_, ax in enumerate(axs):
    panel(ax, k_)
add_sd("SuppFig2", "a", SIMQC)
save(fig, "SuppFig2_simulation_reproduces_GBD_UI")

# S3 VSLY vs GNI
fig, ax = plt.subplots(figsize=(6.5, 4.5))
d = C23[MON]
for g in INC:
    dd = d[d["Income group"] == g]
    ax.scatter(dd["GNI pc, I$"], dd["VSLY, US$"], s=16, color=INC_COL[g], label=g, alpha=.8)
fl = d[d["VSL floor binds"] == True]
ax.scatter(fl["GNI pc, I$"], fl["VSLY, US$"], s=50, facecolor="none", edgecolor="k", lw=.8, label="VSL floor binds")
ax.set(xscale="log", yscale="log", xlabel="GNI per capita, 2023 I$ (PPP)", ylabel="VSLY, 2023 US$")
ax.legend(frameon=False, fontsize=7)
add_sd("SuppFig3", "a", d[["Country", "Income group", "GNI pc, I$", "GNI source 2023", "VSL floor binds", "e(median age)", "VSLY, US$"]])
save(fig, "SuppFig3_VSLY_by_income_2023")

# S4 top-20 countries VLW with UI
fig, ax = plt.subplots(figsize=(7.5, 6))
top = C23[MON].nlargest(20, "VLW, US$ million").iloc[::-1]
ax.barh(top.Country, top["VLW, US$ million"], color=[INC_COL[g] for g in top["Income group"]])
ax.errorbar(top["VLW, US$ million"], top.Country, xerr=[top["VLW, US$ million"] - top["VLW, US$ million lower"],
            top["VLW, US$ million upper"] - top["VLW, US$ million"]], fmt="none", ecolor="k", elinewidth=.7)
ax.set_xlabel("VLW, million 2023 US$ (95% UI)")
ax.tick_params(axis="y", labelsize=7.5)
ax.legend(handles=[Patch(color=INC_COL[g], label=g) for g in INC], frameon=False, loc="lower right")
add_sd("SuppFig4", "a", top)
save(fig, "SuppFig4_top20_countries_VLW_2023")


for name, panels in sdata.items():
    with pd.ExcelWriter(SD / f"{name}_SourceData.xlsx") as xw:
        for p_, df in panels.items():
            df.to_excel(xw, sheet_name=p_, index=False)
    pd.concat([df.assign(panel=p_) for p_, df in panels.items()]).to_csv(SD / f"{name}_SourceData.csv", index=False,
                                                                       encoding="utf-8-sig")


# Tables
def to_md(df, path, title, note, k=2):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "—" if pd.isna(x) else (fmt_int(x) if abs(x) >= 1000 else f"{x:,.{k}f}".replace("-", "−")))
    d = d.fillna("—").astype(str)
    lines = [f"# {title}", "", note, "", "| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(r) + " |" for r in d.values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ui(t, k=None, scale=1):
    return fmt_ui(t[0] / scale, t[1] / scale, t[2] / scale, k)


T1 = []
for g in GN:
    o = G23[g]
    T1.append({"Group": g, "Countries (valued)": f"{o['Countries']} ({o['Countries valued']})",
               "Incident cases": ui(o["Incidence"]), "Deaths": ui(o["Deaths"]),
               "Prevalent cases": ui(o["Prevalence"]),
               "DALYs": ui(o["DALYs"]), "YLLs, % of DALYs": fmt_dec(o["YLLs"][0] / o["DALYs"][0] * 100, 1),
               "Age <5, % of DALYs": fmt_dec(o["Share <5 %"], 1),
               "ASR per 100,000: Incidence": ui(o["Incidence ASR"], 2),
               "ASR per 100,000: Deaths": ui(o["Deaths ASR"], 2),
               "ASR per 100,000: DALYs": ui(o["DALYs ASR"], 1),
               "VLW: million US$": ui(o["VLW"], 1, 1e6),
               "VLW: per child, US$": ui(o["VLW per child"], 2),
               "VLW: % of GDP": ui(o["VLW/GDP %"], 4),
               "VLW: DALY coverage, %": fmt_dec(o["DALY coverage %"], 1)})
T1 = pd.DataFrame(T1)
n_excl = int((~MON).sum())
excl_names = ", ".join(NAME[LIDS[i]] for i in np.where(~MON)[0])
T1_NOTE = (f"GBD 2023 estimates for retinoblastoma (cause 1009) in children younger than 10 years (<5 + 5–9 years), both "
           f"sexes, 130 low- and middle-income countries (World Bank FY2027, fixed cohort). ASR, age-standardised rate within "
           f"<10 years using GBD 2023 standard-population weights for <5 and 5–9 years renormalised to 1 ({W[0]:.4f}, {W[1]:.4f}). "
           f"VLW, value of lost welfare = DALYs × VSLY in 2023 US dollars (US$) at market exchange rates, undiscounted; VSLY = VSL / life "
           f"expectancy at the national median age; VSL = max{{20 × GNI pc, 160 × GNI pc(US) × (GNI pc/GNI pc(US))^1.5}} with GNI in PPP terms, then converted to US$ with each "
           f"country's 2023 price-level ratio (GDP pc in current US$ / GDP pc in current PPP); GDP in the same US$. "
           f"Health measures cover all 130 countries; monetary values exclude {n_excl} countries without income data "
           f"({excl_names}); DALY coverage gives the share of the group's DALYs that are valued. 95% UIs from {ND} draws of "
           f"a split-lognormal distribution fitted to each GBD estimate, fully correlated within a country and independent "
           f"between countries; population, GDP and VSLY held fixed (fully correlated version in Supplementary Table 5).")

T2 = C23[MON].copy()
top15 = set(T2.nlargest(15, "VLW, US$ million").location_id)
q1 = set(T2[T2["Priority quadrant"].str.startswith("Q1")].nlargest(10, "DALYs ASR").location_id)
T2 = T2[T2.location_id.isin(top15 | q1)].sort_values("VLW, US$ million", ascending=False)
T2 = pd.DataFrame({
    "Country": T2.Country, "Income group": T2["Income group"].map(INC_SHORT),
    "Deaths": [fmt_ui(a, b, c) for a, b, c in zip(T2.Deaths, T2["Deaths lower"], T2["Deaths upper"])],
    "DALYs": [fmt_ui(a, b, c) for a, b, c in zip(T2.DALYs, T2["DALYs lower"], T2["DALYs upper"])],
    "DALY ASR per 100,000": [fmt_ui(a, b, c, 1) for a, b, c in zip(T2["DALYs ASR"], T2["DALYs ASR lower"], T2["DALYs ASR upper"])],
    "VLW, million US$": [fmt_ui(a, b, c, 1) for a, b, c in zip(T2["VLW, US$ million"], T2["VLW, US$ million lower"], T2["VLW, US$ million upper"])],
    "VLW/GDP, %": [fmt_dec(x, 4) for x in T2["VLW/GDP, %"]],
    "Rank by VLW": T2["Rank by VLW"].astype(int).astype(str),
    "Selection": np.where(T2.location_id.isin(top15) & T2.location_id.isin(q1), "Top-15 VLW; top-10 Q1 rate",
                         np.where(T2.location_id.isin(top15), "Top-15 VLW", "Top-10 Q1 rate")),
    "Priority quadrant": T2["Priority quadrant"].str.split(" ").str[0]})
T2_NOTE = ("Countries ranked among the 15 largest by total VLW in 2023 and/or among the 10 highest DALY rates within quadrant Q1 of the priority matrix (Fig. 4a; full list in Supplementary Table 1): "
           f"age-standardised DALY rate ≥ the LMIC median ({xmed:.1f} per 100,000) and VLW/GDP ≥ the LMIC median ({ymed:.4f}%). "
           "Q2, high health burden but low VLW/GDP; Q3, low health burden but high VLW/GDP. UIs as in Table 1; "
           "rankings of countries with wide UIs should be read as indicative. LIC, LMC, UMC: low, lower-middle, upper-middle income.")
MAIN_TABLES = {
    "Table1_burden_VLW_LMIC_2023": (T1, "Table 1 | Retinoblastoma burden and value of lost welfare in children <10 years, 130 LMICs, 2023", T1_NOTE),
    "Table2_priority_countries_2023": (T2, "Table 2 | High-burden countries for retinoblastoma policy priority, children <10 years, 2023", T2_NOTE)}
for name, (df, title, note) in MAIN_TABLES.items():
    df.to_csv(TAB_MAIN / f"{name}.csv", index=False, encoding="utf-8-sig")
    to_md(df, TAB_MAIN / f"{name}.md", title, note)
pd.DataFrame([{"file": k_, "title": v[1], "note": v[2]} for k_, v in MAIN_TABLES.items()]).to_csv(
    TAB_MAIN / "table_titles_notes.csv", index=False, encoding="utf-8-sig")

# supplementary tables
S1 = C23.drop(columns=["Rank by VLW"]).sort_values(["Income group", "Country"])
key_years = [1990, 2000, 2010, 2019, 2023]
S2 = TREND[TREND.Year.isin(key_years)].copy()
PCT = []
for g in GN:
    for m in (1, 2, 6):
        for asr in (False, True):
            p_ = pct_change(g, m, asr)
            PCT.append({"Group": g, "Measure": MEAS[m] + (" ASR" if asr else " number"),
                        "Change 1990–2023, %": p_[0], "Lower": p_[1], "Upper": p_[2]})
PCT = pd.DataFrame(PCT)
tv = TREND.set_index(["Group", "Year"])
for g in GN:
    PCT.loc[len(PCT)] = {"Group": g, "Measure": "VLW (each year's VSLY)",
                         "Change 1990–2023, %": (tv.loc[(g, 2023), "VLW"] / tv.loc[(g, 1990), "VLW"] - 1) * 100}
    PCT.loc[len(PCT)] = {"Group": g, "Measure": "VLW (fixed 2023 VSLY)",
                         "Change 1990–2023, %": (tv.loc[(g, 2023), "VLW fixed 2023 VSLY"] / tv.loc[(g, 1990), "VLW fixed 2023 VSLY"] - 1) * 100}
SEX = []
for s_, sl in ((0, "Male"), (1, "Female"), (2, "Both")):
    for g in GMAIN:
        mk = GROUPS[g]
        mm = mk & MON
        pop = POP[s_][:, mk, Y23].sum(axis=1)
        row = {"Sex": sl, "Group": g}
        for m in (1, 2, 6):
            num = N["val"][MI[m], s_][:, mk, Y23].sum(axis=1)
            row[MEAS[m]] = num.sum()
            row[MEAS[m] + " ASR"] = (W * num / pop).sum() * 1e5
        row["VLW, US$ million"] = (N["val"][MI[2], s_][:, mm, Y23].sum(axis=0) * VSLY23[mm]).sum() / 1e6
        SEX.append(row)
SEX = pd.DataFrame(SEX)
SRC = pd.DataFrame([
    {"Input": "GBD 2023 retinoblastoma (cause 1009), 6 measures, <5 and 5–9 y, 130 LMICs, 1990–2023",
     "Source": "GBD Results Tool, version 8352 (R1); including DALYs from the same release",
     "File": "data/R1"},
    {"Input": "Population <5, 5–9", "Source": f"GBD population (POP batch); {N_POP_DERIVED} country-year-age cells derived as Number/Rate",
     "File": "data/POP"},
    {"Input": "Global and regional GBD aggregates", "Source": "GBD Results Tool (R2)", "File": "data/R2"},
    {"Input": "GNI per capita PPP (current), US GDP deflator", "Source": "World Bank WDI (NY.GNP.PCAP.PP.CD, NY.GDP.DEFL.ZS)",
     "File": "reference/wdi_gni_ppp_current.csv, wdi_us_deflator.csv"},
    {"Input": "GDP per capita (PPP, US$), total population", "Source": "World Bank WDI", "File": "reference/wdi_macro_long_1990_2023.csv"},
    {"Input": "Median age, life table e(x)", "Source": "UN WPP 2024", "File": "reference/wpp_median_age.csv, wpp_life_table_both_ex.csv.gz"},
    {"Input": "Standard-population weights", "Source": "GBD 2023 (reconstructed)", "File": "reference/age_weights_gbd2023_reconstructed.csv"},
    {"Input": "Health opportunity cost thresholds", "Source": "Pichon-Riviere 2023; Ochalek 2018", "File": "reference/k_threshold_130.csv"},
    {"Input": "Registry incidence", "Source": "IICC-3 (International Incidence of Childhood Cancer, vol. 3), retinoblastoma table",
     "File": "data/05-retinoblastoma.csv"},
    {"Input": "Previous estimate", "Source": "Previous childhood cancer valuation, Table 10 (0–19 y)", "File": "data/Table_10_Retinoblastoma.csv"},
    {"Input": "Published cost studies", "Source": "Published study abstracts and full texts", "File": "reference/lit_rb_cost.csv"},
    {"Input": "Map", "Source": "df_world2.geojson (China boundary corrected)", "File": "data/df_world2.geojson"},
])
INCOME_FLAGS = pd.DataFrame({"iso3": ISO, "Country": [NAME[l] for l in LIDS], "In monetary set": MON,
                             "GNI 2023 source": FLAG.loc[ISO][2023].values,
                             "Years observed": (FLAG.loc[ISO] == "observed").sum(axis=1).values,
                             "Years imputed": ((FLAG.loc[ISO] != "observed") & (FLAG.loc[ISO] != "missing")).sum(axis=1).values})
SUPP = {
    "SuppTable1_country_results_2023": (S1, "Country results, 2023", T1_NOTE),
    "SuppTable2_trends_key_years": (S2, "Burden and VLW by group, key years 1990–2023", "VLW for countries with complete 1990–2023 income series; fixed-VSLY series holds each country's 2023 VSLY."),
    "SuppTable3_percent_change_1990_2023": (PCT, "Percentage change 1990–2023", "UIs from the same draws in both years (correlated within country)."),
    "SuppTable4_inequality": (INEQ, "Concentration index and slope index of inequality across countries ranked by GNI per capita", "Crude <10-year rates (not age-standardised), weighted by population <10 years; countries ranked by GNI per capita, ties share the mid-rank. Negative CI = concentrated among poorer countries; SII = fitted difference between the richest (rank 1) and poorest (rank 0) end of the income ranking. UIs from the DALY draws (VSLY fixed); not computed for death rate and VLW/GDP."),
    "SuppTable5_UI_assumptions": (UI_TAB, "Uncertainty intervals under independent vs fully correlated country errors, 2023", "VSLY fixed."),
    "SuppTable6_sensitivity": (SENS, "Deterministic sensitivity analyses of 2023 VLW", f"VSL floor binds in {FLOOR_N} countries ({FLOOR_DALY:.1f}% of valued DALYs): {FLOOR_NAMES}."),
    "SuppTable7_registry_vs_GBD_incidence": (REGCMP, "GBD 2023 incidence vs IICC-3 population-based registries, LMICs", "GBD rate at the registry's mid-period year; registries partly inform GBD, so agreement is not fully independent validation."),
    "SuppTable8_published_GBD_comparison": (pd.concat([LANCET_TAB.assign(Table="Published"), PUB.assign(Table="GBD aggregates")], ignore_index=True),
                                            "Comparison with published GBD 2023 figures and GBD global/regional aggregates", "Published: GBD 2023 Cancer Collaborators, Lancet 2025 (all ages, global). Deaths and DALYs agree within 1%; the current Results Tool global <10 incidence exceeds the published all-age incidence, so the published figure comes from a different model round and the incidence row is not a like-for-like check. Aggregate bounds are sums of age-group bounds (approximate, wider than the true UI)."),
    "SuppTable9_bridge_previous_estimate": (BRIDGE, "Bridge from the previous estimate to this study", "The first row is in millions of international dollars. The second row converts to millions of US dollars; its change column is a numerical currency conversion difference. Subsequent changes are in millions of US dollars. Ages 10–19 contribute zero DALYs to the earlier estimate."),
    "SuppTable10_published_cost_vs_model": (LIT, "Published retinoblastoma cost studies vs model VLW per incident case", "PubMed search 2026-11-23 (retinoblastoma AND cost/economic/expenditure terms; 215 records) — studies in cohort countries reporting a per-patient RB cost in money terms. Values checked against abstract or PMC full text (see Verification). Converted to 2023 US$ with the US GDP deflator; price year assumed where not stated."),
    "SuppTable11_sex_specific_2023": (SEX, "Sex-specific results, 2023", "Point estimates."),
    "SuppTable12_mortality_incidence_ratio": (MIR, "Deaths-to-incidence ratio by income group vs clinical survival", "GRBSG 2022 (Lancet Glob Health): 3-year survival 57.3% in low-income countries."),
    "SuppTable13_income_series": (INCOME_FLAGS, "Income series: observed and imputed years", "Missing income values were estimated from within-country GNI/GDP ratios, log-linear interpolation, flat backcasts or one-year carry-forward."),
    "SuppTable14_data_sources": (SRC, "Data sources", ""),
}
with pd.ExcelWriter(TAB_S / "Supplementary_tables.xlsx") as xw:
    for name, (df, title, note) in SUPP.items():
        df.to_csv(TAB_S / f"{name}.csv", index=False, encoding="utf-8-sig")
        to_md(df, TAB_S / f"{name}.md", f"Supplementary Table {name[9:].split('_')[0]} | {title}", note)
        df.to_excel(xw, sheet_name=name[:31], index=False)
