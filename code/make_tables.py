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

CL = {(41.0, 0.20): "Phoenix 41\\degrees C/20\\%",
      (32.0, 0.63): "Miami 32\\degrees C/63\\%",
      (36.0, 0.60): "Heatwave 36\\degrees C/60\\%",
      (30.0, 0.52): "Honolulu 30\\degrees C/52\\%"}

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

# ---- validation: Euler step-size study ----
cv = pd.read_csv(os.path.join(OUT, "validation_convergence.csv"))
rows = [f"{int(r['dt_s'])} & {r['kWh']:.1f} & {r['Tin_max']:.2f} & {r['dev_from_finest']:.2f} \\\\"
        for _, r in cv.iterrows()]
w("t_valid.tex", "\n".join(rows))

# ---- robust optimum selection ----
rows = []
for cln, label in [("hot-arid", "Hot-arid"), ("hot-humid", "Hot-humid")]:
    rt = pd.read_csv(os.path.join(OUT, f"robust_optimum_{cln}.csv"))
    first = True
    for _, r in rt.iterrows():
        lab = label if first else ""
        first = False
        ntr = "none" if str(r["N_trigger"]) == "none" else f"{float(r['N_trigger']):.0f}"
        theta = f"{r['T_on']:.1f} / {r['band']:.1f} / {ntr}"
        opt = f"\\textbf{{{theta}}}" if r["robust_pick"] else theta
        rows.append(f"{lab} & {opt} & {r['kWh_nom']:.1f} & "
                    f"{r['P_feasible']:.2f} & {r['ok_p10']:.2f} \\\\")
    rows.append("\\addlinespace")
w("t_robust.tex", "\n".join(rows))

# ---- Monte-Carlo summary ----
ms = pd.read_csv(os.path.join(OUT, "monte_carlo_summary.csv"))
rows = []
for _, r in ms.iterrows():
    rows.append(
        f"{r['climate']} & {r['save3_med']:.1f} ({r['save3_p10']:.1f}--{r['save3_p90']:.1f}) & "
        f"{100*r['P3_feasible']:.1f}\\% & {100*r['P4_feasible']:.1f}\\% & "
        f"{100*r['P3_wins']:.1f}\\% & {r['e3_med']:.1f} \\\\")
w("t_mc.tex", "\n".join(rows))

# ---- appendix: full bus-stop results (all occupancies) ----
rows = []
CL_SHORT = {(41.0, 0.20): "Phoenix 41\\degrees C/20\\%", (32.0, 0.63): "Miami 32\\degrees C/63\\%",
            (36.0, 0.60): "Heatwave 36\\degrees C/60\\%", (30.0, 0.52): "Honolulu 30\\degrees C/52\\%"}
for (T, RH), label in CL_SHORT.items():
    d = bus[(bus["T_air"] == T) & (bus["RH"] == RH)]
    first = True
    for _, r in d.iterrows():
        lab = label if first else ""
        first = False
        rows.append(f"{lab} & {r['system']} & {int(r['N'])} & {r['relief']:.1f} & "
                    f"{r['n_served']:.0f} & {r['T_perceived']:.1f} & {r['DHR']:.1f} \\\\")
    rows.append("\\addlinespace")
w("t_bus_full.tex", "\n".join(rows))

print("gen tables written to", GEN)
