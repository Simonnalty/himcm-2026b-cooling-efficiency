# -*- coding: utf-8 -*-
"""Generate LaTeX table fragments from output CSVs -> paper/gen/*.tex"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
OUT = os.path.join(ROOT, "output")
GEN = os.path.join(ROOT, "paper", "gen")
os.makedirs(GEN, exist_ok=True)

def w(name, content):
    with open(os.path.join(GEN, name), "w", encoding="utf-8") as fh:
        fh.write(content)

bus = pd.read_csv(os.path.join(OUT, "bus_stop_results.csv"))
sta = pd.read_csv(os.path.join(OUT, "station_policies.csv"))
ven = pd.read_csv(os.path.join(OUT, "venue_scaling.csv"))
eff = {k: pd.read_csv(os.path.join(OUT, f"effectiveness_{k}.csv"))
       for k in ["bus_dry", "bus_humid", "station_arid", "station_humid"]}
tor_d = pd.read_csv(os.path.join(OUT, "tornado_DHR_dry.csv"))
tor_h = pd.read_csv(os.path.join(OUT, "tornado_DHR_humid.csv"))

CL = {(41.0, 0.20): "Hot-arid (Phoenix-type), 41\\degrees C, 20\\% RH",
      (33.0, 0.70): "Hot-humid (Miami-type), 33\\degrees C, 70\\% RH",
      (36.0, 0.60): "Hot-humid heat wave, 36\\degrees C, 60\\% RH",
      (30.0, 0.55): "Warm windy (Honolulu-type), 30\\degrees C, 55\\% RH"}

# ---- bus stop table (N = 12) ----
rows = []
for (T, RH), label in CL.items():
    d = bus[(bus["T_air"] == T) & (bus["RH"] == RH) & (bus["N"] == 12)]
    first = True
    for _, r in d.iterrows():
        lab = label if first else ""
        first = False
        rows.append(
            f"{lab} & {r['system']} & {r['dT_sys']:.1f} & {r['n_served']:.0f} & "
            f"{r['relief']:.1f} & {r['T_perceived']:.1f} & {r['DHR']:.1f} & "
            f"{r['safe']} \\\\")
    rows.append("\\addlinespace")
w("t_bus.tex", "\n".join(rows))

# ---- station policy table ----
rows = []
for cl, label in [("hot-arid", "Hot-arid"), ("hot-humid", "Hot-humid")]:
    d = sta[sta["climate"] == cl]
    first = True
    for _, r in d.iterrows():
        lab = label if first else ""
        first = False
        feas = "yes" if r["ok_share"] >= 0.8 else "no"
        dhr_s = "--" if pd.isna(r["DHR"]) else f"{r['DHR']:.1f}"
        rows.append(
            f"{lab} & {r['policy']} & {r['energy_kWh']:.0f} & {r['mean_relief_C']:.1f} & "
            f"{dhr_s} & {r['strain_pdegH']:.0f} & {100*r['ok_share']:.0f}\\% & "
            f"{r['cost_per_day_usd']:.1f} & {feas} \\\\")
    rows.append("\\addlinespace")
w("t_station.tex", "\n".join(rows))

# ---- effectiveness table ----
rows = []
panel = {"bus_dry": "Bus stop, hot-arid", "bus_humid": "Bus stop, hot-humid",
         "station_arid": "Station, hot-arid", "station_humid": "Station, hot-humid"}
best = {}
for k, df in eff.items():
    df_f = df[df["feasible"]]
    if len(df_f):
        best[k] = df_f.loc[df_f["E"].idxmax(), "option"]
for k, df in eff.items():
    first = True
    for _, r in df.iterrows():
        lab = panel[k] if first else ""
        first = False
        feas = "yes" if r["feasible"] else "no"
        opt = r["option"]
        b = r["option"] == best.get(k)
        opt_s = f"\\textbf{{{opt}}}" if b else opt
        rows.append(
            f"{lab} & {opt_s} & {feas} & {r['perf']:.2f} & {r['exp']:.2f} & "
            f"{r['energy']:.2f} & {r['cost']:.2f} & {r['E']:.3f} \\\\")
    rows.append("\\addlinespace")
w("t_eff.tex", "\n".join(rows))

# ---- venue table ----
VORD = ["Phoenix stadium concourse (hot-arid, dense)",
        "Guangzhou rail waiting hall (hot-humid, semi-open)",
        "Singapore outdoor patio (hot-humid, open-air)",
        "Honolulu airport pavilion (moderate, windy)"]
VL = {"Phoenix stadium concourse (hot-arid, dense)": "Stadium concourse, Phoenix (hot-arid, dense)",
      "Guangzhou rail waiting hall (hot-humid, semi-open)": "Rail hall, Guangzhou (hot-humid, semi-open)",
      "Singapore outdoor patio (hot-humid, open-air)": "Outdoor patio, Singapore (hot-humid, open-air)",
      "Honolulu airport pavilion (moderate, windy)": "Airport pavilion, Honolulu (warm, windy)"}
rows = []
for v in VORD:
    d = ven[ven["venue"] == v].set_index("system")
    cells = []
    for s in ["Fans", "Medium evaporative", "Large evaporative"]:
        r = d.loc[s]
        star = "$^\\ast$" if r["safe"] == "yes" else "$^\\dagger$"
        cells.append(f"{r['DHR']:.0f}{star} ({r['relief']:.1f}\\degrees C)")
    rows.append(f"{VL[v]} & " + " & ".join(cells) + " \\\\")
w("t_venue.tex", "\n".join(rows))

# ---- tornado table (dry vs humid side by side) ----
td = tor_d.set_index("param"); th = tor_h.set_index("param")
rows = []
for p in ["Occupancy N", "Rated drop dT_rated", "Electric power P",
          "Relative humidity RH", "Air temperature T", "Saturation effectiveness",
          "Coverage area A"]:
    rd, rh = td.loc[p], th.loc[p]
    label = p.replace('dT_rated', r'$\Delta T_{\mathrm{rated}}$') \
             .replace('Saturation effectiveness', r'Saturation effectiveness $\eta$')
    rows.append(f"{label} & "
                f"{rd['low']:.0f} / {rd['high']:.0f} & {rd['swing']:.1f} & "
                f"{rh['low']:.0f} / {rh['high']:.0f} & {rh['swing']:.1f} \\\\")
w("t_tornado.tex", "\n".join(rows))

print("gen tables written to", GEN)
