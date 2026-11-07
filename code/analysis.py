# -*- coding: utf-8 -*-
"""
2026 HiMCM Problem B - "Air Conditioning the Environment"
Cooling efficiency & effectiveness for open-air venues and semi-open public spaces.

Physics core + bus-stop model (Req 2a) + rail-station transient model (Req 2b)
+ venue scaling (Req 3) + multi-objective effectiveness (Req 4).

Run:  python analysis.py
Outputs: ../output/*.csv, ../figures/*.png, ../output/summary.json
All console output is ASCII (Windows-safe).
"""
import os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(ROOT, "output");  os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
    "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.30, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})
COL = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728",
       "P0": "#7f7f7f", "P1": "#1f77b4", "P2": "#2ca02c", "P3": "#d62728", "P4": "#9467bd"}

# =====================================================================
# 1. THERMODYNAMIC CORE
# =====================================================================
P_ATM = 101.325          # kPa, atmospheric pressure
RHO_CP = 1.2 * 1005.0    # J/(m^3 K) volumetric heat capacity of air
ELEC_PRICE = 0.14        # USD/kWh (sensitivity-varied later)
ETA_EVAP = 0.85          # evaporative saturation effectiveness
COMFORT_T = 27.0         # perceived degC: no further relief is valued below this
T_TOLERABLE = 30.0       # perceived degC: "warm but acceptable" ceiling
T_STRAIN = 38.0          # perceived degC: heat-strain flag
WBGT_SAFE = 32.0         # WBGT limit for light activity (waiting)

def esat(T):
    """Saturation vapour pressure (Tetens), kPa."""
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))

def wet_bulb(T, RH):
    """Psychrometric wet-bulb temperature (degC) via bisection.
    T: dry-bulb degC, RH: relative humidity fraction 0-1."""
    e = RH * esat(T)
    lo, hi = -20.0, T
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = esat(mid) - 0.00066 * P_ATM * (T - mid) * (1 + 0.00115 * mid) - e
        if f > 0:   # f is increasing in mid: root lies below mid
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def evap_dT(T, RH, rated, eta=ETA_EVAP):
    """Achievable air-temperature drop of an evaporative cooler:
    capped by the rated value and by the wet-bulb depression."""
    return float(min(rated, eta * (T - wet_bulb(T, RH))))

def fan_factor(RH):
    """Perceived-benefit scaling of a fan with humidity
    (sweat evaporation is less effective when the air is near saturation)."""
    return float(1.0 - 0.7 * max(0.0, (RH - 0.4) / 0.6))

def fan_dT(T, RH, rated=3.0):
    """Fan: perceived ('feels like') cooling of a served person, degC."""
    return rated * fan_factor(RH)

def relief_cap(T_env):
    """Cooling nobody needs is not useful: relief is capped by the excess
    heat above the comfort ceiling."""
    return max(0.0, T_env - COMFORT_T)

def effective_relief(dT_sys, T_env):
    return max(0.0, min(dT_sys, relief_cap(T_env)))

def wbgt_eff(T_p, Tw, d_rad=1.5):
    """Effective WBGT for a shaded, light-activity person experiencing
    perceived temperature T_p (fan offset included via T_p)."""
    return 0.7 * Tw + 0.3 * (T_p + d_rad)

def strain_exposure(WB, T_p):
    """Heat-strain index (deg): WBGT excess above 32 or perceived excess above 38."""
    return max(0.0, WB - WBGT_SAFE) + max(0.0, T_p - T_STRAIN)

# =====================================================================
# 2. REQUIREMENT 2a - BUS STOP MODEL
# =====================================================================
BUS_SYS = {
    "A. Fan":               dict(power=0.25, area=6.0,  rated=3.0, kind="fan",  capex=250.0),
    "B. Evaporative":       dict(power=1.2,  area=12.0, rated=5.0, kind="evap", capex=900.0),
    "C. Large Evaporative": dict(power=2.5,  area=22.0, rated=8.0, kind="evap", capex=2200.0),
}
AREA_PERSON = 0.75   # m^2 per waiting person
BUS_HOURS = 14.0     # operating hours per day
LIFE_YEARS = 6.0     # equipment lifetime for amortization

# Calibrated against station normals (July, afternoon conditions):
#   Phoenix  41 degC / 20%  (NWS Phoenix, NOAA normals)
#   Miami    32 degC / 63%  (Florida Climate Center: July 4pm RH 59-63%)
#   Honolulu 30 degC / 52%  (2pm RH ~52%)
CLIMATES = {
    "Dry extreme (Phoenix 41C/20%)":    dict(T=41.0, RH=0.20),
    "Humid typical (Miami 32C/63%)":    dict(T=32.0, RH=0.63),
    "Humid heatwave (36C/60%)":         dict(T=36.0, RH=0.60),
    "Moderate windy (Honolulu 30C/52%)":dict(T=30.0, RH=0.52),
}
OCCUPANCIES = {"light N=6": 6, "peak N=12": 12, "crush N=20": 20}

def bus_stop_row(sysname, s, T, RH, N):
    if s["kind"] == "fan":
        dT = fan_dT(T, RH, s["rated"])
        Tw = wet_bulb(T, RH)
    else:
        dT = evap_dT(T, RH, s["rated"])
        Tw = wet_bulb(T, RH)
    n_served = min(N, s["area"] / AREA_PERSON)
    rel = effective_relief(dT, T)
    benefit = n_served * rel                      # person-degC per operating hour
    cei = benefit / s["power"] if s["power"] > 0 else np.nan   # DHR: person-degC-h / kWh
    T_p = T - rel                                  # perceived temp of a served person
    WB = wbgt_eff(T_p, Tw)
    daily_cost = s["power"] * BUS_HOURS * ELEC_PRICE + s["capex"] / (LIFE_YEARS * 365.0)
    return dict(system=sysname, kind=s["kind"], T_air=T, RH=RH, N=N,
                dT_sys=round(dT, 2), n_served=round(n_served, 1),
                relief=round(rel, 2), T_perceived=round(T_p, 1),
                WBGT_eff=round(WB, 1),
                safe="yes" if (WB <= WBGT_SAFE and T_p < T_STRAIN) else "NO",
                benefit_per_h=round(benefit, 1),
                DHR=round(cei, 1),
                cost_per_pdh_usd=round(daily_cost / benefit, 4) if benefit > 0 else np.nan)

rows = []
for cname, c in CLIMATES.items():
    for oname, N in OCCUPANCIES.items():
        for sysname, s in BUS_SYS.items():
            rows.append(bus_stop_row(sysname, s, c["T"], c["RH"], N))
bus_df = pd.DataFrame(rows)
bus_df["climate"] = bus_df["T_air"].astype(str) + "C/" + (bus_df["RH"] * 100).astype(int).astype(str) + "%"
bus_df.to_csv(os.path.join(OUT, "bus_stop_results.csv"), index=False)

# ---- thermodynamic (COP-like) view for the discussion in Req 1 ----
thermo = []
for sysname, s in BUS_SYS.items():
    if s["kind"] == "evap":
        flow = 3000.0 if s["power"] < 2.0 else 6000.0   # m3/h nominal airflow
        Q_th = RHO_CP * (flow / 3600.0) * s["rated"]     # W thermal at rated drop
        thermo.append(dict(system=sysname, thermal_kW=Q_th / 1000.0,
                           ratio=round(Q_th / 1000.0 / s["power"], 1)))
    else:
        # fan: extra convective+evaporative heat loss from a person, ~40 W each
        Q_th = 8 * 40.0
        thermo.append(dict(system=sysname, thermal_kW=Q_th / 1000.0,
                           ratio=round(Q_th / 1000.0 / s["power"], 1)))
thermo_df = pd.DataFrame(thermo)
thermo_df.to_csv(os.path.join(OUT, "thermodynamic_view.csv"), index=False)

# =====================================================================
# 3. REQUIREMENT 2b - SEMI-OPEN RAIL STATION (transient zone model)
# =====================================================================
STATION = dict(
    V=1920.0,          # m3  (40 x 12 x 4 m waiting hall)
    UA=180.0,          # W/K envelope conduction (roof+walls, louvered front)
    ACH0=3.0,          # baseline stack/leakage air changes per hour
    kw=2.0,            # additional ACH per m/s of wind
    wind=1.5,          # m/s mean wind
    kappa=4.0,         # thermal-mass factor (air + structure, multiples of air)
    S_max=4000.0,      # W solar gain through the open/louvered front at peak
    q_sens=55.0,       # W sensible heat per person
    base_N=25.0,       # baseline occupancy
    surge=70.0,        # passengers added per train arrival pulse
    tau_surge=240.0,   # s, decay time of a surge pulse
    n_units=4, unit_power=2.5, unit_flow=6000.0, rated=8.0, eta=0.85, recirc=0.8,
    n_fans=10, fan_power=0.06,
    n_ac=2, ac_cool=12.0, cop=3.2, ac_recirc=0.65,
    d_rad=1.5,         # degC globe-vs-air radiant offset (shaded)
    dwb_in=0.4,        # degC wet-bulb rise inside (latent gains)
)

def diurnal(h, lo, hi):
    """Half-sine diurnal profile: lo at 06:00, hi at 15:00, back to lo at 23:00."""
    h = h % 24.0
    if 6.0 <= h <= 15.0:
        f = 0.5 - 0.5 * math.cos(math.pi * (h - 6.0) / 9.0)
    elif 15.0 < h <= 23.0:
        f = 0.5 + 0.5 * math.cos(math.pi * (h - 15.0) / 8.0)
    else:
        f = 0.0
    return lo + (hi - lo) * f

def train_arrivals(h0=6.0, h1=22.0):
    evs, h = [], h0
    while h < h1:
        evs.append(h)
        peak = (7.0 <= h < 9.0) or (17.0 <= h < 19.0)
        h += 12.0 / 60.0 if peak else 20.0 / 60.0
    return np.array(evs)

def occupancy(t_h, ev, P):
    N = np.full_like(t_h, P["base_N"], dtype=float)
    tau_h = P["tau_surge"] / 3600.0          # t_h is in HOURS
    for a in ev:
        dt = t_h - a
        m = dt >= 0
        N[m] += P["surge"] * np.exp(-dt[m] / tau_h)
    return np.minimum(N, 140.0)

def climate_arrays(Tlo, Thi, RHmin, RHmax, P=STATION, dt=30.0):
    """Policy-independent exogenous arrays, computed once per climate/parameter set."""
    t_s = np.arange(6 * 3600, 22 * 3600, dt)
    t_h = t_s / 3600.0
    f = np.array([diurnal(h, 0.0, 1.0) for h in t_h])
    Tout = np.array([diurnal(h, Tlo, Thi) for h in t_h])
    RH = RHmax - (RHmax - RHmin) * f
    Tw_out = np.array([wet_bulb(t, r) for t, r in zip(Tout, RH)])
    N = occupancy(t_h, train_arrivals(), P)
    S = P["S_max"] * f
    ACH = P["ACH0"] + P["kw"] * P["wind"]
    Gv = RHO_CP * P["V"] * ACH / 3600.0
    C = RHO_CP * P["V"] * P["kappa"]
    dTe = np.minimum(P["rated"], P["eta"] * (Tout - Tw_out))  # achievable drop
    return dict(t_h=t_h, dt=dt, Tout=Tout, RH=RH, Tw_out=Tw_out, N=N, S=S,
                Gv=Gv, C=C, dTe=dTe, P=P)

def run_policy(pre, policy):
    """Sequential Euler integration under one operating policy."""
    P = pre["P"]; dt = pre["dt"]
    t_h, Tout, RH, Tw_out = pre["t_h"], pre["Tout"], pre["RH"], pre["Tw_out"]
    N, S, Gv, C, dTe = pre["N"], pre["S"], pre["Gv"], pre["C"], pre["dTe"]
    n = len(t_h)
    Tin = np.empty(n); Tin[0] = Tout[0] - 0.5
    Tp = np.empty(n); WB = np.empty(n)
    Tp[0] = Tin[0]
    WB[0] = 0.7 * (Tw_out[0] + P["dwb_in"]) + 0.3 * (Tp[0] + P["d_rad"])
    fan_on = np.zeros(n, bool); evap_on = np.zeros(n, bool); ac_on = np.zeros(n, bool)
    evap_state = False          # bang-bang controller state with hysteresis
    kWh = 0.0

    for i in range(1, n):
        h = t_h[i]
        fan_on[i] = policy.get("fan", False)
        ac_on[i] = policy.get("ac", False)
        evp_mode = policy.get("evap", "off")
        if evp_mode == "always":
            evap_on[i] = True
        elif evp_mode == "auto":
            # thermostat control: hold perceived temp near the acceptability
            # ceiling; pre-cool before the morning/evening surge waves
            T_on, band, Ntrig = policy.get("theta", (30.0, 2.5, 70.0))
            pre_ok = (6.5 <= h <= 7.6) or (16.4 <= h <= 17.6)
            if pre_ok or (Ntrig is not None and N[i] > Ntrig):
                evap_state = True
            elif evap_state:
                if Tp[i - 1] < T_on - band:
                    evap_state = False
            else:
                if Tp[i - 1] > T_on:
                    evap_state = True
            evap_on[i] = evap_state

        Qc = 0.0
        if evap_on[i]:
            Qc += P["n_units"] * RHO_CP * (P["unit_flow"] / 3600.0) * dTe[i] * P["recirc"]
        if ac_on[i]:
            Qc += P["n_ac"] * P["ac_cool"] * 1000.0 * P["ac_recirc"]

        dT = Tout[i] - Tin[i - 1]
        Qgain = P["UA"] * dT + S[i] + N[i] * P["q_sens"] + Gv * dT
        Tin[i] = Tin[i - 1] + (Qgain - Qc) * dt / C

        fan_off = fan_dT(Tout[i], RH[i], 3.0) * 0.8 if fan_on[i] else 0.0
        Tp[i] = Tin[i] - fan_off
        # wet-bulb shift inside: evaporative adds moisture, refrigerative dehumidifies
        dwb = P["dwb_in"] + (0.2 if evap_on[i] else 0.0) - (1.4 if ac_on[i] else 0.0)
        WB[i] = 0.7 * (Tw_out[i] + dwb) + 0.3 * (Tp[i] + P["d_rad"])

        kWh += (P["n_fans"] * P["fan_power"] * fan_on[i]
                + P["n_units"] * P["unit_power"] * evap_on[i]
                + P["n_ac"] * P["ac_cool"] / P["cop"] * ac_on[i]) * dt / 3600.0

    return dict(t_h=t_h, Tout=Tout, Tin=Tin, Tp=Tp, WB=WB, N=N, RH=RH,
                fan=fan_on, evap=evap_on, ac=ac_on, kWh=kWh)

def simulate_station(Tlo, Thi, RHmin, RHmax, policy, P=STATION, dt=30.0):
    return run_policy(climate_arrays(Tlo, Thi, RHmin, RHmax, P=P, dt=dt), policy)

POLICIES = {
    "P0 No cooling":   dict(),
    "P1 Fans only":    dict(fan=True),
    "P2 Evap always":  dict(fan=True, evap="always"),
    "P3 Hybrid auto":  dict(fan=True, evap="auto"),
    "P4 Refrigerative":dict(fan=True, ac=True),
}
# Design-day envelopes calibrated to July normals:
#   hot-arid  = Phoenix (low 28, high 41, RH 20% afternoon / 45% morning)
#   hot-humid = Guangzhou (low 28, high 33, RH 60% afternoon / 88% morning)
STATION_CLIMATES = {
    "hot-arid":   dict(Tlo=28.0, Thi=41.0, RHmin=0.20, RHmax=0.45),
    "hot-humid":  dict(Tlo=28.0, Thi=33.0, RHmin=0.60, RHmax=0.88),
}

station_runs = {}
station_rows = []
for cl_name, cl in STATION_CLIMATES.items():
    base = simulate_station(**cl, policy=POLICIES["P0 No cooling"])
    for pname, pol in POLICIES.items():
        r = simulate_station(**cl, policy=pol)
        station_runs[(cl_name, pname)] = r
        # t_h is in HOURS, so trapezoid integrals are already person-degC-hours
        person_h = float(np.trapezoid(r["N"], r["t_h"]))
        relief = float(np.trapezoid(r["N"] * (base["Tp"] - r["Tp"]), r["t_h"]))
        strain = float(np.trapezoid(
            r["N"] * np.array([strain_exposure(w, t) for w, t in zip(r["WB"], r["Tp"])]),
            r["t_h"]))
        ok_share = float(np.trapezoid(
            r["N"] * ((r["Tp"] <= T_TOLERABLE) & (r["WB"] <= WBGT_SAFE)), r["t_h"])) / person_h
        mean_relief = relief / person_h
        capex = 4000.0 + (10000.0 if pol.get("evap") else 0.0) + (7000.0 if pol.get("ac") else 0.0)
        daily_cost = r["kWh"] * ELEC_PRICE + capex / (7.0 * 365.0)
        station_rows.append(dict(climate=cl_name, policy=pname,
                                 energy_kWh=round(r["kWh"], 1),
                                 person_hours=round(person_h, 0),
                                 relief_pdegH=round(relief, 0),
                                 mean_relief_C=round(mean_relief, 2),
                                 DHR=round(relief / r["kWh"], 1) if r["kWh"] > 0 else np.nan,
                                 strain_pdegH=round(strain, 0),
                                 ok_share=round(ok_share, 3),
                                 cost_per_day_usd=round(daily_cost, 1),
                                 cost_per_pdegH_usd=round(daily_cost / relief, 4) if relief > 0 else np.nan,
                                 T_in_max=round(float(np.max(r["Tin"])), 1),
                                 WBGT_max=round(float(np.max(r["WB"])), 1)))
station_df = pd.DataFrame(station_rows)
station_df.to_csv(os.path.join(OUT, "station_policies.csv"), index=False)

# =====================================================================
# 3.5 MODEL VALIDATION + CONTROL OPTIMISATION + UNCERTAINTY (V2)
# =====================================================================
def metrics_vs_base(pre, base, r, T_acc=30.0):
    """relief / strain / acceptable-share of one run against its own baseline."""
    relief = float(np.trapezoid(pre["N"] * (base["Tp"] - r["Tp"]), pre["t_h"]))
    strain = float(np.trapezoid(
        pre["N"] * (np.maximum(0.0, r["WB"] - WBGT_SAFE) + np.maximum(0.0, r["Tp"] - T_STRAIN)),
        pre["t_h"]))
    ok = float(np.trapezoid(
        pre["N"] * ((r["Tp"] <= T_acc) & (r["WB"] <= WBGT_SAFE)), pre["t_h"])) / \
        float(np.trapezoid(pre["N"], pre["t_h"]))
    return relief, strain, ok

# ---- (a) numerical convergence: Euler step-size independence ----
conv = []
for dt in [120.0, 60.0, 30.0, 15.0, 7.5]:
    r = simulate_station(27.0, 40.0, 0.15, 0.35, POLICIES["P3 Hybrid auto"], dt=dt)
    conv.append(dict(dt_s=dt, kWh=round(r["kWh"], 3), Tin_max=round(float(np.max(r["Tin"])), 3),
                     strain=round(float(np.trapezoid(
                         r["N"] * (np.maximum(0.0, r["WB"] - WBGT_SAFE) + np.maximum(0.0, r["Tp"] - T_STRAIN)),
                         r["t_h"])), 2)))
conv_df = pd.DataFrame(conv)
finest = conv_df.iloc[-1]
conv_df["dev_from_finest"] = (conv_df["kWh"] - finest["kWh"]).abs().round(3)
conv_df.to_csv(os.path.join(OUT, "validation_convergence.csv"), index=False)

# ---- (b) steady-state energy-balance check (constant forcing vs analytic) ----
pre_c = climate_arrays(38.0, 38.0, 0.15, 0.15)
pre_c["S"][:] = 3000.0
pre_c["N"][:] = 25.0
Tw_c = pre_c["Tw_out"][0]
dTe_c = min(STATION["rated"], STATION["eta"] * (38.0 - Tw_c))
Qc_c = STATION["n_units"] * RHO_CP * (STATION["unit_flow"] / 3600.0) * dTe_c * STATION["recirc"]
G_c = pre_c["Gv"] + STATION["UA"]
T_eq = 38.0 - (Qc_c - 3000.0 - 25.0 * STATION["q_sens"]) / G_c
r_c = run_policy(pre_c, POLICIES["P2 Evap always"])
T_end = float(r_c["Tin"][-1])
ebal = pd.DataFrame([dict(quant=["analytic equilibrium Tin", "simulated final Tin", "abs. error"],
                          value=[round(T_eq, 3), round(T_end, 3), round(abs(T_end - T_eq), 3)])])
ebal.to_csv(os.path.join(OUT, "validation_energy_balance.csv"), index=False)

# ---- (c) control-policy optimisation: grid search over thermostat parameters ----
def optimise_station(cl, T_ons, bands, Ntrigs):
    pre = climate_arrays(**cl)
    base = run_policy(pre, {})
    N_int = float(np.trapezoid(pre["N"], pre["t_h"]))
    _, base_strain, _ = metrics_vs_base(pre, base, run_policy(pre, {}))
    rows = []
    for T_on in T_ons:
        for band in bands:
            for Ntrig in Ntrigs:
                pol = dict(fan=True, evap="auto", theta=(T_on, band, Ntrig))
                r = run_policy(pre, pol)
                rel, strain, ok = metrics_vs_base(pre, base, r)
                rows.append(dict(T_on=T_on, band=band, N_trigger=("none" if Ntrig is None else Ntrig),
                                 kWh=round(r["kWh"], 1), relief_pdegH=round(rel, 0),
                                 strain=round(strain, 1), ok_share=round(ok, 3),
                                 feasible=(ok >= 0.8 and strain <= 1.0)))
    g = pd.DataFrame(rows)
    g["capex_day"] = (4000.0 + 10000.0) / (7.0 * 365.0)
    g["cost_day"] = (g["kWh"] * ELEC_PRICE + g["capex_day"]).round(2)
    feas = g[g["feasible"]].sort_values(["kWh", "strain"]).reset_index(drop=True)
    g["rank"] = np.nan
    if len(feas):
        g.loc[feas.index[:len(feas)], "rank"] = np.arange(1, len(feas) + 1)
    nom = g[(g["T_on"] == 30.0) & (g["band"] == 2.5) & (g["N_trigger"] == 70.0)].iloc[0]
    return g, feas, nom

T_ONS = [28.5, 29.0, 29.5, 30.0, 30.5, 31.0]
BANDS = [2.0, 2.5, 3.0, 3.5]
NTRIGS = [70.0, 90.0, None]
opt_results, opt_feas, opt_nom = {}, {}, {}
for cln, clv in STATION_CLIMATES.items():
    g, feas, nom = optimise_station(clv, T_ONS, BANDS, NTRIGS)
    opt_results[cln] = g
    opt_feas[cln] = feas
    opt_nom[cln] = nom
    g.to_csv(os.path.join(OUT, f"optimisation_{cln}.csv"), index=False)
opt_best = {c: opt_feas[c].iloc[0].to_dict() for c in opt_feas if len(opt_feas[c])}

# ---- (c2) ROBUST selection: candidates re-scored under plant uncertainty ----
# A setpoint that rides the service floor on the nominal model may dip below it
# under parameter uncertainty. Robust rule: P(ok_share >= 0.8) >= 0.95 over draws.
def mc_ok_for_theta(cl, theta, n=150, seed=11):
    rg = np.random.default_rng(seed)
    oks = []
    for _ in range(n):
        Pm = dict(STATION)
        Pm["UA"] = STATION["UA"] * max(0.5, rg.normal(1.0, 0.15))
        Pm["kappa"] = rg.uniform(3.0, 5.5)
        Pm["kw"] = STATION["kw"] * rg.uniform(0.7, 1.4)
        Pm["q_sens"] = STATION["q_sens"] * max(0.5, rg.normal(1.0, 0.12))
        Pm["surge"] = STATION["surge"] * max(0.5, rg.normal(1.0, 0.12))
        Pm["eta"] = rg.uniform(0.78, 0.90)
        Pm["recirc"] = rg.uniform(0.72, 0.88)
        T_acc = rg.uniform(28.5, 31.5)
        pre_m = climate_arrays(**cl, P=Pm)
        base = run_policy(pre_m, {})
        r3 = run_policy(pre_m, dict(fan=True, evap="auto", theta=theta))
        _, _, ok3 = metrics_vs_base(pre_m, base, r3, T_acc)
        oks.append(ok3)
    return np.array(oks)

robust_opt = {}
for cln, clv in STATION_CLIMATES.items():
    g = opt_results[cln]
    feas = g[g["feasible"]].sort_values(["kWh", "strain"])
    cands = feas.head(5)[["T_on", "band", "N_trigger", "kWh"]].to_dict("records")
    trig = feas[feas["N_trigger"].astype(str) == "70.0"].head(3)[
        ["T_on", "band", "N_trigger", "kWh"]].to_dict("records")
    cands.extend(trig)
    cands.append(dict(T_on=30.0, band=2.5, N_trigger=70.0, kWh=float(opt_nom[cln]["kWh"])))
    seen, uniq = set(), []
    for c in cands:
        key = (c["T_on"], c["band"], str(c["N_trigger"]))
        if key not in seen:
            seen.add(key); uniq.append(c)
    cands = uniq
    rows = []
    for c in cands:
        theta = (c["T_on"], c["band"], None if c["N_trigger"] == "none" else float(c["N_trigger"]))
        oks = mc_ok_for_theta(clv, theta, n=150)
        rows.append(dict(T_on=c["T_on"], band=c["band"], N_trigger=c["N_trigger"],
                         kWh_nom=round(c["kWh"], 1), P_feasible=round(float((oks >= 0.8).mean()), 3),
                         ok_p10=round(float(np.percentile(oks, 10)), 3)))
    rtab = pd.DataFrame(rows)
    ok_r = rtab[rtab["P_feasible"] >= 0.95].sort_values("kWh_nom")
    pick = ok_r.iloc[0].to_dict() if len(ok_r) else rtab.sort_values(["P_feasible", "kWh_nom"]).iloc[0].to_dict()
    rtab["robust_pick"] = (rtab["T_on"] == pick["T_on"]) & (rtab["band"] == pick["band"]) & \
                          (rtab["N_trigger"] == pick["N_trigger"])
    robust_opt[cln] = dict(table=rtab.to_dict("records"), pick=pick)
    rtab.to_csv(os.path.join(OUT, f"robust_optimum_{cln}.csv"), index=False)

# ---- (d) Monte-Carlo uncertainty quantification (400 plant/parameter draws) ----
rng = np.random.default_rng(7)
N_MC = 400
mc_rows = []
for cln, clv in STATION_CLIMATES.items():
    recs = []
    for d in range(N_MC):
        Pm = dict(STATION)
        Pm["UA"] = STATION["UA"] * max(0.5, rng.normal(1.0, 0.15))
        Pm["kappa"] = rng.uniform(3.0, 5.5)
        Pm["kw"] = STATION["kw"] * rng.uniform(0.7, 1.4)      # ACH = 3 + 2v, +/-40%
        Pm["q_sens"] = STATION["q_sens"] * max(0.5, rng.normal(1.0, 0.12))
        Pm["surge"] = STATION["surge"] * max(0.5, rng.normal(1.0, 0.12))
        Pm["eta"] = rng.uniform(0.78, 0.90)
        Pm["recirc"] = rng.uniform(0.72, 0.88)
        T_acc = rng.uniform(28.5, 31.5)
        pre_m = climate_arrays(**clv, P=Pm)
        base = run_policy(pre_m, {})
        r2 = run_policy(pre_m, dict(fan=True, evap="always"))
        r3 = run_policy(pre_m, dict(fan=True, evap="auto"))
        r4 = run_policy(pre_m, dict(fan=True, ac=True))
        m2, s2, o2 = metrics_vs_base(pre_m, base, r2, T_acc)
        m3, s3, o3 = metrics_vs_base(pre_m, base, r3, T_acc)
        m4, s4, o4 = metrics_vs_base(pre_m, base, r4, T_acc)
        recs.append(dict(draw=d, e2=r2["kWh"], e3=r3["kWh"], e4=r4["kWh"],
                         ok2=o2, ok3=o3, ok4=o4, strain2=s2, strain3=s3, strain4=s4,
                         rel2=m2, rel3=m3, rel4=m4))
    mc = pd.DataFrame(recs)
    mc["save3_vs2_pct"] = 100.0 * (1.0 - mc["e3"] / mc["e2"])
    mc["clim"] = cln
    mc_rows.append(mc)
mc_all = pd.concat(mc_rows, ignore_index=True)
mc_all.to_csv(os.path.join(OUT, "monte_carlo_draws.csv"), index=False)

def pct(s, q):
    return float(np.percentile(s, q))
mc_summary = []
for cln in STATION_CLIMATES:
    m = mc_all[mc_all["clim"] == cln]
    feas3 = m["ok3"] >= 0.8
    feas4 = m["ok4"] >= 0.8
    feas2 = m["ok2"] >= 0.8
    # P3 = lowest-energy feasible option among {P2, P3, P4}
    e_min_other = np.where(feas2, m["e2"], np.inf)
    e_min_other = np.minimum(e_min_other, np.where(feas4, m["e4"], np.inf))
    win3 = feas3 & (m["e3"] <= e_min_other + 1e-9)
    mc_summary.append(dict(
        climate=cln, draws=N_MC,
        save3_med=round(pct(m["save3_vs2_pct"], 50), 1),
        save3_p10=round(pct(m["save3_vs2_pct"], 10), 1),
        save3_p90=round(pct(m["save3_vs2_pct"], 90), 1),
        ok3_med=round(pct(m["ok3"], 50), 3), ok3_p10=round(pct(m["ok3"], 10), 3),
        P3_feasible=round(float(feas3.mean()), 3),
        P4_feasible=round(float(feas4.mean()), 3),
        P2_feasible=round(float(feas2.mean()), 3),
        P3_wins=round(float(win3.mean()), 3),
        e2_med=round(pct(m["e2"], 50), 1), e3_med=round(pct(m["e3"], 50), 1)))
mc_summary = pd.DataFrame(mc_summary)
mc_summary.to_csv(os.path.join(OUT, "monte_carlo_summary.csv"), index=False)

# ---- (e) carbon accounting (EPA eGRID-style regional intensities) ----
CARBON = {"hot-arid": 0.49, "hot-humid": 0.52}   # kg CO2e / kWh (eGRID subregional)
SEASON_DAYS = 120
carbon_rows = []
for cln in STATION_CLIMATES:
    d = station_df[station_df["climate"] == cln]
    for _, r in d.iterrows():
        carbon_rows.append(dict(climate=cln, policy=r["policy"], kWh_day=r["energy_kWh"],
                                kgCO2_day=round(r["energy_kWh"] * CARBON[cln], 1),
                                tCO2_season=round(r["energy_kWh"] * CARBON[cln] * SEASON_DAYS / 1000.0, 2)))
carbon_df = pd.DataFrame(carbon_rows)
carbon_df.to_csv(os.path.join(OUT, "station_carbon.csv"), index=False)

# =====================================================================
# 4. REQUIREMENT 3 - VENUE SCALING (4 venues x 3 generic systems)
# =====================================================================
# Generic systems at venue scale: name -> (electric kW, rated drop, coverage m2, kind)
GEN_SYS = {
    "Fans":                dict(power_kWpm2=0.004, rated=3.0, area=200.0, kind="fan"),
    "Medium evaporative":  dict(power_kWpm2=0.030, rated=5.0, area=400.0, kind="evap"),
    "Large evaporative":   dict(power_kWpm2=0.075, rated=8.0, area=700.0, kind="evap"),
}
VENUES = {
    "Phoenix stadium concourse (hot-arid, dense)":  dict(T=40.0, RH=0.20, N=6000, area=3000.0, density=2.0),
    "Guangzhou rail waiting hall (hot-humid, semi-open)": dict(T=33.0, RH=0.75, N=800, area=1500.0, density=1.0),
    "Singapore outdoor patio (hot-humid, open-air)": dict(T=31.5, RH=0.64, N=120, area=300.0, density=0.8),
    "Honolulu airport pavilion (moderate, windy)":  dict(T=30.0, RH=0.52, N=250, area=600.0, density=0.6),
}
venue_rows = []
for vname, v in VENUES.items():
    for sname, s in GEN_SYS.items():
        if s["kind"] == "fan":
            dT = fan_dT(v["T"], v["RH"], s["rated"])
        else:
            dT = evap_dT(v["T"], v["RH"], s["rated"])
        n_served = min(v["N"], s["area"] * v["density"])
        rel = effective_relief(dT, v["T"])
        power = s["power_kWpm2"] * s["area"]
        benefit = n_served * rel
        T_p = v["T"] - rel
        WB = wbgt_eff(T_p, wet_bulb(v["T"], v["RH"]))
        venue_rows.append(dict(venue=vname, system=sname, T_air=v["T"], RH=v["RH"],
                               dT_sys=round(dT, 2), n_served=round(n_served, 0),
                               relief=round(rel, 2), power_kW=round(power, 2),
                               benefit_pdegH_h=round(benefit, 0),
                               DHR=round(benefit / power, 1) if power > 0 else np.nan,
                               safe="yes" if (WB <= WBGT_SAFE and T_p < T_STRAIN) else "NO"))
venue_df = pd.DataFrame(venue_rows)
venue_df.to_csv(os.path.join(OUT, "venue_scaling.csv"), index=False)

# =====================================================================
# 5. REQUIREMENT 4 - EFFECTIVENESS (multi-objective)
# =====================================================================
W = dict(perf=0.30, exp=0.25, energy=0.25, cost=0.20)   # default weights
OK_FLOOR = 0.80   # hard service floor: >=80% of person-time in acceptable conditions

def norm_clip(x, x_max):
    return float(np.clip(1.0 - x / x_max, 0.0, 1.0)) if x_max > 0 else 1.0

def effectiveness_scores(kind, climate=None):
    out = []
    if kind == "bus":
        df = bus_df[(bus_df["N"] == 12) & (bus_df["T_air"] == climate["T"])]
        for _, r in df.iterrows():
            perf = min(1.0, r["relief"] / 8.0)
            exp_s = float(np.clip((T_STRAIN - r["T_perceived"]) / 8.0, 0.0, 1.0))
            daily = BUS_SYS[r["system"]]["power"] * BUS_HOURS * ELEC_PRICE + \
                    BUS_SYS[r["system"]]["capex"] / (LIFE_YEARS * 365.0)
            en = norm_clip(daily, 8.0)
            co = norm_clip(daily, 8.0)
            E = W["perf"] * perf + W["exp"] * exp_s + W["energy"] * en + W["cost"] * co
            out.append(dict(option=r["system"], feasible=(r["safe"] == "yes"),
                            perf=round(perf, 2), exp=round(exp_s, 2),
                            energy=round(en, 2), cost=round(co, 2), E=round(E, 3)))
    else:  # station policies
        df = station_df[station_df["climate"] == climate]
        e_max = df["energy_kWh"].max()
        c_max = df["cost_per_day_usd"].max()
        for _, r in df.iterrows():
            perf = min(1.0, r["mean_relief_C"] / 8.0)
            exp_s = r["ok_share"]
            en = norm_clip(r["energy_kWh"], e_max) if e_max > 0 else 1.0
            co = norm_clip(r["cost_per_day_usd"], c_max) if c_max > 0 else 1.0
            E = W["perf"] * perf + W["exp"] * exp_s + W["energy"] * en + W["cost"] * co
            out.append(dict(option=r["policy"], feasible=(r["ok_share"] >= OK_FLOOR),
                            perf=round(perf, 2), exp=round(exp_s, 2),
                            energy=round(en, 2), cost=round(co, 2), E=round(E, 3)))
    return pd.DataFrame(out)

eff_frames = {
    "bus_dry": effectiveness_scores("bus", CLIMATES["Dry extreme (Phoenix 41C/20%)"]),
    "bus_humid": effectiveness_scores("bus", CLIMATES["Humid typical (Miami 32C/63%)"]),
    "station_arid": effectiveness_scores("station", "hot-arid"),
    "station_humid": effectiveness_scores("station", "hot-humid"),
}
for k, df in eff_frames.items():
    df.to_csv(os.path.join(OUT, f"effectiveness_{k}.csv"), index=False)

# ---- weight-robustness: random Dirichlet weights among FEASIBLE options ----
rng = np.random.default_rng(42)
def weight_robustness(df, n=400):
    df_f = df[df["feasible"]]
    if len(df_f) == 0:
        return {}
    wins = {o: 0 for o in df_f["option"]}
    comps = df_f[["perf", "exp", "energy", "cost"]].values
    opts = df_f["option"].values
    for _ in range(n):
        w = rng.dirichlet([2, 2, 2, 2])
        scores = comps @ w
        wins[opts[int(np.argmax(scores))]] += 1
    return wins
robust = {k: weight_robustness(df) for k, df in eff_frames.items() if len(df) > 1}
pd.DataFrame(robust).to_csv(os.path.join(OUT, "weight_robustness.csv"))

# ---- sensitivity tornado: DHR of system B, dry vs humid heat ----
def dhr_B(T=41.0, RH=0.20, N=12, rated=5.0, power=1.2, area=12.0, eta=ETA_EVAP):
    dT = evap_dT(T, RH, rated, eta)
    n_s = min(N, area / AREA_PERSON)
    rel = effective_relief(dT, T)
    return n_s * rel / power

def tornado_rows(T, RH, Nbase):
    base_kw = dict(T=T, RH=RH, N=Nbase)
    base = dhr_B(**base_kw)
    rows = []
    for label, kw_lo, kw_hi in [
        ("Air temperature T", dict(T=T - 3.0), dict(T=T + 3.0)),
        ("Relative humidity RH", dict(RH=max(0.05, RH - 0.08)), dict(RH=min(0.95, RH + 0.08))),
        ("Occupancy N", dict(N=max(1, Nbase - 4)), dict(N=Nbase + 4)),
        ("Rated drop dT_rated", dict(rated=3.5), dict(rated=6.5)),
        ("Electric power P", dict(power=0.9), dict(power=1.5)),
        ("Coverage area A", dict(area=8.0), dict(area=16.0)),
        ("Saturation effectiveness", dict(eta=0.75), dict(eta=0.92)),
    ]:
        lo = dhr_B(**{**base_kw, **kw_lo}); hi = dhr_B(**{**base_kw, **kw_hi})
        rows.append(dict(param=label, low=round(lo, 1), high=round(hi, 1),
                         base=round(base, 1), swing=round(abs(hi - lo), 1)))
    return pd.DataFrame(rows).sort_values("swing", ascending=False), base

torn_df, torn_base_dry = tornado_rows(41.0, 0.20, 12)
torn_df_humid, torn_base_humid = tornado_rows(33.0, 0.70, 12)
torn_df.to_csv(os.path.join(OUT, "tornado_DHR_dry.csv"), index=False)
torn_df_humid.to_csv(os.path.join(OUT, "tornado_DHR_humid.csv"), index=False)

# =====================================================================
# 6. FIGURES
# =====================================================================
def fig1_schematic():
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 6); ax.axis("off")
    ax.add_patch(Rectangle((1.5, 1.2), 7.5, 3.6, fill=False, lw=2.2, ec="k"))
    ax.text(5.25, 5.15, "Control volume: shelter / waiting hall (semi-open)", ha="center", fontsize=11)
    ax.add_patch(Rectangle((3.2, 1.2), 4.0, 1.0, fc="#a6cee3", alpha=0.7))
    ax.text(5.2, 1.7, "occupied zone (people)", ha="center", fontsize=9)
    ax.text(9.35, 3.9, "open boundary:\nair exchange $\\dot{V}=\\lambda V$", fontsize=9, va="top")
    ax.annotate("", xy=(9.9, 3.0), xytext=(9.05, 3.0),
                arrowprops=dict(arrowstyle="<->", lw=1.6))
    ax.annotate("", xy=(1.5, 3.0), xytext=(0.6, 3.0),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#e31a1c"))
    ax.text(0.45, 3.25, "$Q_{solar}$", color="#e31a1c", fontsize=10)
    ax.annotate("", xy=(1.9, 1.15), xytext=(1.9, 0.62),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#b8860b"))
    ax.text(2.08, 0.62, "$Q_{ground}$", color="#b8860b", fontsize=10)
    ax.annotate("", xy=(4.6, 3.1), xytext=(4.6, 3.85),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f77b4"))
    ax.text(3.1, 3.95, "$N q_{p}$  (people)", color="#1f77b4", fontsize=10)
    ax.annotate("", xy=(6.6, 2.35), xytext=(6.6, 3.2),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#2ca02c"))
    ax.text(6.75, 3.3, "$Q_{cool}$", color="#2ca02c", fontsize=10)
    ax.text(6.95, 1.75, "useful jet reaches people;\nrest escapes to ambient", fontsize=8)
    ax.annotate("", xy=(11.6, 2.4), xytext=(10.0, 2.4),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#808080"))
    ax.text(10.8, 2.62, "lost cooling\n(dissipation)", color="#808080", fontsize=8, ha="center")
    ax.text(5.4, 0.18, "Efficiency = useful cooling delivered at the person / energy consumed",
            ha="center", fontsize=10, style="italic")
    fig.savefig(os.path.join(FIG, "fig1_control_volume.png")); plt.close(fig)

def fig2_response():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    ax = axes[0]
    dbd = np.linspace(0, 18, 100)
    for key, rated, colr in [("B", 5.0, COL["B"]), ("C", 8.0, COL["C"])]:
        ys = [min(rated, ETA_EVAP * d) for d in dbd]
        ax.plot(dbd, ys, color=colr, lw=2, label=f"System {key} (rated {rated:.0f}$^\\circ$C)")
    ax.axvline(4.6, ls="--", c="gray", lw=1)
    ax.annotate("Miami-type humid\ndepression $\\approx$ 4.6 K", xy=(4.6, 4.2), fontsize=8,
                xytext=(6.2, 1.6), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("Phoenix-type dry\ndepression $\\approx$ 19 K", xy=(18, 8), fontsize=8, ha="right")
    ax.set_xlabel("Wet-bulb depression $T_{db}-T_{wb}$ (K)")
    ax.set_ylabel("Actual air-temperature drop (degC)")
    ax.set_title("(a) Evaporative systems: climate-limited")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    rh = np.linspace(0.2, 1.0, 100)
    ax.plot(rh * 100, [fan_dT(33.0, r) for r in rh], color=COL["A"], lw=2)
    ax.set_xlabel("Relative humidity (%)")
    ax.set_ylabel("Perceived cooling (degC)")
    ax.set_title("(b) Fan system: humidity-limited")
    ax.annotate("full benefit below ~40% RH", xy=(40, 3.0), xytext=(52, 2.3), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=1))
    fig.savefig(os.path.join(FIG, "fig2_system_response.png")); plt.close(fig)

def fig3_bus():
    clis = list(CLIMATES.keys())
    short = ["Phoenix\n41C/20%", "Miami\n33C/70%", "Heatwave\n36C/60%", "Honolulu\n30C/55%"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, N, title in zip(axes, [12, 20], ["(a) peak occupancy N=12", "(b) crush occupancy N=20"]):
        x = np.arange(len(clis)); w = 0.26
        for k, (sname, colr) in enumerate(zip(BUS_SYS.keys(), [COL["A"], COL["B"], COL["C"]])):
            vals = [bus_df[(bus_df["system"] == sname) & (bus_df["T_air"] == c["T"]) &
                           (bus_df["N"] == N)]["DHR"].iloc[0] for c in CLIMATES.values()]
            ax.bar(x + (k - 1) * w, vals, w, color=colr, label=sname)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
        ax.set_ylabel("DHR (person-$^\\circ$C$\\cdot$h / kWh)")
        ax.set_title(title); ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "fig3_bus_DHR.png")); plt.close(fig)

def fig4_station_timeseries():
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), sharex=True)
    for row, cl in enumerate(["hot-arid", "hot-humid"]):
        r0 = station_runs[(cl, "P0 No cooling")]
        # 15-min rolling mean: pulses to ~95 persons every 12-20 min train arrival
        Ns = pd.Series(r0["N"]).rolling(30, center=True, min_periods=1).mean()
        axes[row, 0].plot(r0["t_h"], Ns, color="k", lw=1.8)
        axes[row, 0].set_ylim(0, 110)
        axes[row, 0].text(0.03, 0.92, "pulses to ~95 persons per train arrival",
                          transform=axes[row, 0].transAxes, fontsize=7, va="top")
        axes[row, 0].set_ylabel("occupancy N (persons)")
        axes[row, 0].set_title(f"{'(a-d) ' if row==0 else ''}Occupancy ({cl}, 15-min mean)")
        for pname, colr in [("P0 No cooling", COL["P0"]), ("P1 Fans only", COL["P1"]),
                            ("P2 Evap always", COL["P2"]), ("P3 Hybrid auto", COL["P3"]),
                            ("P4 Refrigerative", COL["P4"])]:
            r = station_runs[(cl, pname)]
            axes[row, 1].plot(r["t_h"], r["Tin"], color=colr, lw=1.5, label=pname)
            axes[row, 2].plot(r["t_h"], r["WB"], color=colr, lw=1.5)
        axes[row, 1].plot(r0["t_h"], r0["Tout"], color="k", lw=1.2, ls="--", label="$T_{out}$")
        axes[row, 2].axhline(WBGT_SAFE, color="r", ls="--", lw=1.2)
        axes[row, 2].text(6.2, WBGT_SAFE + 0.15, "WBGT safe limit 32", color="r", fontsize=8)
        axes[row, 1].set_ylabel("$T_{in}$ (degC)")
        axes[row, 2].set_ylabel("WBGT$_{eff}$ (degC)")
    axes[0, 1].legend(fontsize=7, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel("hour of day")
    fig.savefig(os.path.join(FIG, "fig4_station_timeseries.png")); plt.close(fig)

def fig5_station_bars():
    pols = list(POLICIES.keys())
    x = np.arange(len(pols))
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.4), sharex=True)
    for row, cl in enumerate(["hot-arid", "hot-humid"]):
        df = station_df[station_df["climate"] == cl].set_index("policy").loc[pols]
        cols = [COL[p.split()[0]] for p in pols]
        ax = axes[row, 0]
        ax.bar(x, df["energy_kWh"], 0.62, color=cols)
        ax.set_ylabel("energy (kWh/day)")
        ax.set_title(f"{cl}: electricity")
        for i, v in enumerate(df["energy_kWh"]):
            ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=8)
        ax2 = axes[row, 1]
        ax2.bar(x, df["strain_pdegH"], 0.62, color=cols, hatch="//", edgecolor="k")
        ax2.set_ylabel("person-degC$\\cdot$h above safe limits")
        ax2.set_title(f"{cl}: heat-strain exposure")
        ymax = max(float(df["strain_pdegH"].max()), 1.0)
        ax2.set_ylim(0, ymax * 1.3 + 2)
        if float(df["strain_pdegH"].max()) == 0.0:
            ax2.text(0.5, 0.55, "no exceedance of safe limits", transform=ax2.transAxes,
                     ha="center", fontsize=9, color="#555555")
        for i, v in enumerate(df["strain_pdegH"]):
            ax2.text(i, v + ymax * 0.03, f"{v:.0f}", ha="center", fontsize=8)
    for ax in axes[1]:
        ax.set_xticks(x); ax.set_xticklabels([p.split(" ")[0] for p in pols])
    fig.savefig(os.path.join(FIG, "fig5_policy_bars.png")); plt.close(fig)

def fig6_venues():
    piv = venue_df.pivot(index="venue", columns="system", values="DHR")
    order = list(VENUES.keys()); sys_order = ["Fans", "Medium evaporative", "Large evaporative"]
    piv = piv.loc[order, sys_order]
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(sys_order))); ax.set_xticklabels(sys_order, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o.split("(")[0].strip() for o in order], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.0f}", ha="center", va="center",
                    color="white" if piv.values[i, j] > piv.values.max() * 0.6 else "k", fontsize=9)
    fig.colorbar(im, ax=ax, label="DHR (person-$^\\circ$C$\\cdot$h/kWh)")
    ax.set_title("Cooling efficiency across venues x systems (model scales unchanged)")
    ax.grid(False)
    fig.savefig(os.path.join(FIG, "fig6_venue_heatmap.png")); plt.close(fig)

def fig7_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax, cl, title in zip(axes, ["hot-arid", "hot-humid"], ["(a) station, hot-arid", "(b) station, hot-humid"]):
        df = station_df[station_df["climate"] == cl]
        for _, r in df.iterrows():
            ax.scatter(r["cost_per_day_usd"], r["mean_relief_C"], s=90,
                       color=COL[r["policy"].split()[0]], zorder=3)
            ax.annotate(r["policy"].split(" ")[0] + f" ({r['mean_relief_C']:.1f}C)",
                        (r["cost_per_day_usd"], r["mean_relief_C"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        pts = df[["cost_per_day_usd", "mean_relief_C"]].values
        order = np.argsort(pts[:, 0])
        ax.plot(pts[order, 0], pts[order, 1], "k--", lw=1, alpha=0.6)
        ax.set_xlabel("total cost (USD/day, amortised + energy)")
        ax.set_ylabel("mean relief per person (degC)")
        ax.set_title(title)
    fig.savefig(os.path.join(FIG, "fig7_pareto.png")); plt.close(fig)

def fig8_tornado():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, df, base, title in [
            (axes[0], torn_df, torn_base_dry, "(a) hot-arid (41$^\\circ$C / 20% RH)"),
            (axes[1], torn_df_humid, torn_base_humid, "(b) hot-humid (33$^\\circ$C / 70% RH)")]:
        y = np.arange(len(df))
        for i, (_, r) in enumerate(df.iterrows()):
            lo, hi = min(r["low"], r["high"]), max(r["low"], r["high"])
            ax.barh(i, hi - base, left=base, color="#2ca02c", alpha=0.85)
            ax.barh(i, lo - base, left=base, color="#d62728", alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(df["param"], fontsize=8)
        ax.axvline(base, color="k", lw=1.4)
        ax.set_xlabel("DHR of system B (person-$^\\circ$C$\\cdot$h/kWh)")
        ax.set_title(title)
    fig.suptitle("One-at-a-time sensitivity (green: high value, red: low value)", y=1.02)
    fig.savefig(os.path.join(FIG, "fig8_tornado.png")); plt.close(fig)

def fig9_weights():
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    labels = list(robust["station_humid"].keys())
    vals = [robust["station_humid"][k] for k in labels]
    cols = [COL[k.split()[0]] for k in labels]
    ax.bar(range(len(labels)), vals, color=cols)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.split(" ")[0] for l in labels])
    ax.set_ylabel("times ranked #1 (of 400 random weight sets)")
    ax.set_title("Robustness of the effectiveness ranking to objective weights (station, hot-humid)")
    fig.savefig(os.path.join(FIG, "fig9_weights.png")); plt.close(fig)

def fig0_framework():
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    layers = [
        ("Inputs", "climate profiles (NOAA normals) · venue geometry\nfleet sizes & tariff / carbon intensity", "#f0f0f0"),
        ("Layer 1 - Physics", "psychrometric wet bulb · evaporative / fan / refrigerative\nlimits · transient zone heat balance (30-s Euler)", "#dbe9f6"),
        ("Layer 2 - People", "perceived temperature · comfort cap 27$^\\circ$C\nacceptable 30$^\\circ$C · WBGT$_{eff}$ heat strain", "#ddeddd"),
        ("Layer 3 - Efficiency", "DHR = person-$^\\circ$C$\\cdot$h relief / kWh\ncost per delivered relief · CO$_2$ per season", "#fbe7c8"),
        ("Layer 4 - Effectiveness", "hard service floor (80% acceptable) · weighted utility\nPareto frontier · grid-search optimisation · Monte-Carlo", "#f4d3d3"),
        ("Decisions", "fleet selection · thermostat setpoints · tiered\noperations · extreme-heat playbook", "#e8e0f0"),
    ]
    y = 11.6
    H = 1.45
    for i, (t, sub, c) in enumerate(layers):
        ax.add_patch(Rectangle((0.8, y - H), 8.4, H, fc=c, ec="k", lw=1.2))
        ax.text(5.0, y - 0.38, t, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(5.0, y - 0.98, sub, ha="center", va="center", fontsize=8)
        if i < len(layers) - 1:
            ax.annotate("", xy=(5.0, y - H - 0.38), xytext=(5.0, y - H),
                        arrowprops=dict(arrowstyle="->", lw=1.8))
        y -= 1.92
    ax.text(9.7, 9.6, "feedback:\nsensitivity &\nuncertainty\nre-open layer 1", fontsize=7.5,
            ha="center", va="top", color="#555555")
    ax.annotate("", xy=(9.55, 9.55), xytext=(9.55, 4.1),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#888888",
                                connectionstyle="arc3,rad=-0.35"))
    fig.savefig(os.path.join(FIG, "fig0_framework.png")); plt.close(fig)

def fig10_opt_surface():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for ax, cln, title in zip(axes, ["hot-arid", "hot-humid"],
                              ["(a) hot-arid", "(b) hot-humid"]):
        g = opt_results[cln]
        g70 = g[g["N_trigger"].astype(str) == "70.0"]
        piv = g70.pivot_table(index="band", columns="T_on", values="kWh")
        feas = g70.pivot_table(index="band", columns="T_on", values="feasible")
        im = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap="YlGnBu", shading="nearest")
        fy, fx = np.where(feas.values.astype(bool))
        ax.scatter(piv.columns[fx], piv.index[fy], marker="s", s=28, c="#2ca02c", zorder=3)
        if len(opt_feas[cln]):
            b = opt_feas[cln].iloc[0]
            ax.scatter([b["T_on"]], [b["band"]], marker="*", s=260, c="red", zorder=4,
                       label="grid optimum")
        ax.scatter([30.0], [2.5], marker="o", s=90, facecolor="white", edgecolor="k",
                   zorder=4, label="nominal rule")
        fig.colorbar(im, ax=ax, label="energy (kWh/day)")
        ax.set_xlabel("thermostat ON threshold $T_{on}$ (degC perceived)")
        ax.set_ylabel("hysteresis band (degC)")
        ax.set_title(title); ax.legend(fontsize=7, loc="upper right")
    fig.savefig(os.path.join(FIG, "fig10_opt_surface.png")); plt.close(fig)

def fig11_mc():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    m_ar = mc_all[mc_all["clim"] == "hot-arid"]
    axes[0].hist(m_ar["save3_vs2_pct"], bins=36, color="#1f77b4", alpha=0.75)
    nom = 100 * (1 - station_df[(station_df["climate"] == "hot-arid") &
                                (station_df["policy"] == "P3 Hybrid auto")]["energy_kWh"].iloc[0] /
                 station_df[(station_df["climate"] == "hot-arid") &
                            (station_df["policy"] == "P2 Evap always")]["energy_kWh"].iloc[0])
    axes[0].axvline(nom, color="r", ls="--", lw=1.5)
    axes[0].text(nom, axes[0].get_ylim()[1] * 0.92, f" nominal {nom:.0f}%", color="r", fontsize=8)
    axes[0].set_xlabel("hybrid energy saving vs always-on (%)")
    axes[0].set_ylabel("draws (of 400)")
    axes[0].set_title("(a) hot-arid: saving distribution")
    m_hu = mc_all[mc_all["clim"] == "hot-humid"]
    for col, lab, colr in [("ok3", "P3 hybrid", COL["P3"]), ("ok4", "P4 refrigerative", COL["P4"])]:
        xs = np.sort(m_hu[col]); ys = np.arange(1, len(xs) + 1) / len(xs)
        axes[1].step(xs, ys, where="post", color=colr, lw=1.8, label=lab)
    axes[1].axvline(0.8, color="k", ls="--", lw=1.2)
    axes[1].text(0.815, 0.06, "service floor 0.80", fontsize=8)
    axes[1].set_xlabel("acceptable share of person-time")
    axes[1].set_ylabel("cumulative probability")
    axes[1].set_xlim(0.3, 1.02); axes[1].set_title("(b) hot-humid: feasibility under plant uncertainty")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.savefig(os.path.join(FIG, "fig11_mc.png")); plt.close(fig)

def fig12_regime():
    Ts = np.arange(28.0, 44.1, 0.5)
    RHs = np.arange(0.15, 0.91, 0.05)
    winners = np.zeros((len(RHs), len(Ts)))   # 0 none-safe, 1 fan, 2 evap B, 3 evap C
    for i, RH in enumerate(RHs):
        for j, T in enumerate(Ts):
            scores = {}
            for k, (nm, s) in enumerate(BUS_SYS.items(), start=1):
                dT = fan_dT(T, RH, s["rated"]) if s["kind"] == "fan" else evap_dT(T, RH, s["rated"])
                rel = effective_relief(dT, T)
                T_p = T - rel
                WB = wbgt_eff(T_p, wet_bulb(T, RH))
                safe = (WB <= WBGT_SAFE) and (T_p < T_STRAIN)
                if safe and rel > 0:
                    scores[k] = min(12.0, s["area"] / AREA_PERSON) * rel / s["power"]
            if scores:
                winners[i, j] = max(scores, key=scores.get)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#c0c0c0", "#1f77b4", "#ff7f0e", "#2ca02c"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    pc = ax.pcolormesh(Ts, RHs * 100, winners, cmap=cmap, norm=norm, shading="nearest")
    # wet-bulb-depression contour where system B reaches its rated cap
    D, R = np.meshgrid(Ts, RHs)
    dbd = D - np.array([[wet_bulb(t, r) for t in Ts] for r in RHs])
    ax.contour(D, R * 100, dbd, levels=[5.88], colors="k", linestyles="--", linewidths=1.2)
    ax.text(29.2, 77, "evap. B hits rated cap\n(dashed: DBD = 5.9 K)", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#c0c0c0", label="no system keeps person below strain"),
                       Patch(fc="#1f77b4", label="fan wins (best feasible DHR)"),
                       Patch(fc="#ff7f0e", label="medium evaporative wins"),
                       Patch(fc="#2ca02c", label="large evaporative wins")],
              fontsize=8, loc="upper left", framealpha=0.95)
    ax.set_xlabel("dry-bulb temperature (degC)")
    ax.set_ylabel("relative humidity (%)")
    ax.set_title("Technology regime map (bus stop, N=12): best feasible system by climate")
    fig.savefig(os.path.join(FIG, "fig12_regime.png")); plt.close(fig)

fig0_framework(); fig1_schematic(); fig2_response(); fig3_bus(); fig4_station_timeseries()
fig5_station_bars(); fig6_venues(); fig7_pareto(); fig8_tornado(); fig9_weights()
fig10_opt_surface(); fig11_mc(); fig12_regime()

# =====================================================================
# 7. SUMMARY
# =====================================================================
summary = {
    "bus_stop_peak_N12": bus_df[bus_df["N"] == 12][
        ["system", "T_air", "RH", "dT_sys", "relief", "DHR", "T_perceived", "safe"]
    ].to_dict("records"),
    "thermo_view": thermo_df.to_dict("records"),
    "station": station_df.to_dict("records"),
    "venues_best_DHR": {v: venue_df[venue_df["venue"] == v].sort_values("DHR", ascending=False).iloc[0][
        ["system", "DHR", "relief", "safe"]].to_dict() for v in VENUES},
    "effectiveness_station_humid": eff_frames["station_humid"].to_dict("records"),
    "weight_robustness_station_humid": robust["station_humid"],
    "tornado": torn_df.to_dict("records"),
    "validation_convergence": conv_df.to_dict("records"),
    "energy_balance": dict(analytic=float(T_eq), simulated=float(T_end),
                           abs_err=round(abs(T_end - T_eq), 3)),
    "optimisation_best": opt_best,
    "optimisation_nominal": {c: opt_nom[c][["kWh", "ok_share", "strain"]].to_dict()
                             for c in opt_nom},
    "monte_carlo": mc_summary.to_dict("records"),
    "carbon": carbon_df.to_dict("records"),
}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=1, default=float)

print("=" * 72)
print("BUS STOP (peak N=12): DHR = person-degC-h per kWh")
print(bus_df[bus_df["N"] == 12].pivot_table(index="system", columns="T_air", values="DHR"))
print("\nStation policies:")
print(station_df[["climate", "policy", "energy_kWh", "mean_relief_C", "DHR",
                  "strain_pdegH", "ok_share", "cost_per_day_usd", "T_in_max", "WBGT_max"]]
      .to_string(index=False))
print("\nVenue scaling (best system per venue):")
for v, d in summary["venues_best_DHR"].items():
    print(f"  {v[:45]:47s} -> {d['system']:20s} DHR={d['DHR']:.0f} relief={d['relief']}C safe={d['safe']}")
print("\nEffectiveness (station, hot-humid):")
print(eff_frames["station_humid"].to_string(index=False))
print("\nWeight robustness (station hot-humid):", robust["station_humid"])
print("\nValidation (Euler step convergence, P3 hot-arid):")
print(conv_df.to_string(index=False))
print(f"Energy balance: analytic {T_eq:.3f} vs simulated {T_end:.3f} degC "
      f"(err {abs(T_end - T_eq):.3f})")
print("\nControl optimisation (best feasible per climate):")
for c, b in opt_best.items():
    n = opt_nom[c]
    print(f"  {c}: T_on={b['T_on']} band={b['band']} Ntrig={b['N_trigger']} "
          f"kWh={b['kWh']} ok={b['ok_share']}  |  nominal rule kWh={n['kWh']} ok={n['ok_share']}")
print("\nMonte Carlo (400 draws):")
print(mc_summary.to_string(index=False))
print("\nCarbon (kg CO2e/kWh, eGRID-style):", CARBON)
print(carbon_df.to_string(index=False))
print("\nAll figures + CSVs written. DONE.")
