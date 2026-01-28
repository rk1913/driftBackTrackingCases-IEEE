import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

# ============================================================
# 0) EDIT ONLY THESE PATHS
# ============================================================
PHASE1_PATH = "/Users/sampathrajana/Desktop/phase1_drifter_plus_currents.csv"
WIND_NC_PATH = "/Users/sampathrajana/Documents/winds.nc"

# ============================================================
# 1) CONFIG
# ============================================================
R = 6371e3
DT_HOURS = 6
dt = DT_HOURS * 3600

SEED = 42
np.random.seed(SEED)

# Forward rollout horizons (endpoint errors)
ROLLOUT_HOURS = [6, 24, 48, 72]   # <-- includes 48h
ROLLOUT_SAMPLES_PER_H = 400
ROLLOUT_ATTEMPT_MULT = 25

# Backtracking
HORIZONS_HOURS = [48, 72]
EPS_LIST_KM = [20, 10, 5]
N_CANDIDATES = 3000
BOX_KM = 150
N_TRIALS = 25

# ============================================================
# 2) HELPERS
# ============================================================
def meters_per_degree_lat():
    return np.pi * R / 180.0

def meters_per_degree_lon(lat_deg):
    return np.pi * R * np.cos(np.radians(lat_deg)) / 180.0

def lon_wrap(lon):
    return ((lon + 180) % 360) - 180

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return (R*c)/1000.0

def summarize_err(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mae_km": np.nan, "rmse_km": np.nan, "median_km": np.nan,
                "p90_km": np.nan, "p95_km": np.nan, "mean_km": np.nan}
    return {
        "n": int(arr.size),
        "mae_km": float(np.mean(np.abs(arr))),
        "rmse_km": float(np.sqrt(np.mean(arr**2))),
        "median_km": float(np.median(arr)),
        "p90_km": float(np.quantile(arr, 0.90)),
        "p95_km": float(np.quantile(arr, 0.95)),
        "mean_km": float(np.mean(arr)),
    }

def skill(mae_model, mae_phys):
    if (mae_phys is None) or (not np.isfinite(mae_phys)) or mae_phys <= 1e-12:
        return np.nan
    return 1.0 - (mae_model / mae_phys)

def ecdf(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1) / len(x) if len(x) else np.array([])
    return x, y

# ============================================================
# 3) LOAD PHASE1
# ============================================================
df = pd.read_csv(PHASE1_PATH, parse_dates=["time"])
df = df.sort_values(["drifter_id", "time"]).reset_index(drop=True)

print("Drifters:", df["drifter_id"].nunique())
print("Time span:", df["time"].min(), "->", df["time"].max())

# ============================================================
# 4) LOAD WIND + ATTACH u10/v10 (with diagnostics)
# ============================================================
wds = xr.open_dataset(WIND_NC_PATH)
if "valid_time" in wds.coords:
    wds = wds.rename({"valid_time": "time"})

df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
wds["time"] = pd.to_datetime(wds["time"].values)

t  = xr.DataArray(df["time"].values, dims="points")
la = xr.DataArray(df["latitude"].values, dims="points")
lo = xr.DataArray(df["longitude"].values, dims="points")

w_time = wds[["u10", "v10"]].sel(time=t, method="nearest", tolerance=np.timedelta64(3, "h"))
w_pts = w_time.sel(latitude=la, longitude=lo, method="nearest")

df["u10"] = w_pts["u10"].values
df["v10"] = w_pts["v10"].values

df["wind_missing"] = (df["u10"].isna() | df["v10"].isna()).astype(int)
df["u10"] = df["u10"].fillna(0.0)
df["v10"] = df["v10"].fillna(0.0)

print("Wind attached. Missing wind rows:", int(df["wind_missing"].sum()))

w_lat = wds["latitude"].values
w_lon = wds["longitude"].values
lat_idx = np.abs(w_lat[:, None] - df["latitude"].values[None, :]).argmin(axis=0)
lon_idx = np.abs(w_lon[:, None] - df["longitude"].values[None, :]).argmin(axis=0)
sel_lat = w_lat[lat_idx]
sel_lon = w_lon[lon_idx]
grid_dist_km = haversine_km(df["latitude"].values, df["longitude"].values, sel_lat, sel_lon)
print("Wind-grid nearest distance (km):",
      {"mean": float(np.mean(grid_dist_km)), "p90": float(np.quantile(grid_dist_km, 0.90))})

# ============================================================
# 5) BUILD 6h STEP DATASET
# ============================================================
df["lat_next"]  = df.groupby("drifter_id")["latitude"].shift(-1)
df["lon_next"]  = df.groupby("drifter_id")["longitude"].shift(-1)
df["time_next"] = df.groupby("drifter_id")["time"].shift(-1)

step = df.dropna(subset=["lat_next", "lon_next", "time_next"]).copy()
step = step[(step["time_next"] - step["time"]).dt.total_seconds() == dt].copy()

step["v_true_n"] = (step["lat_next"] - step["latitude"]) * meters_per_degree_lat() / dt
step["v_true_e"] = (step["lon_next"] - step["longitude"]) * meters_per_degree_lon(step["latitude"]) / dt

groups = step["drifter_id"].to_numpy()

# ============================================================
# 6) TUNE WIND LEEWAY alpha (train-only)
# ============================================================
def tune_alpha(train_df, alphas):
    best_alpha = 0.0
    best_mean = np.inf
    lat0 = train_df["latitude"].to_numpy()
    lon0 = train_df["longitude"].to_numpy()

    for a in alphas:
        u_phys = train_df["uo"].to_numpy() + a * train_df["u10"].to_numpy()
        v_phys = train_df["vo"].to_numpy() + a * train_df["v10"].to_numpy()

        lat_p = lat0 + (v_phys*dt)/R*(180/np.pi)
        lon_p = lon0 + (u_phys*dt)/(R*np.cos(np.radians(lat0)))*(180/np.pi)
        lon_p = lon_wrap(lon_p)

        err = haversine_km(lat_p, lon_p,
                           train_df["lat_next"].to_numpy(),
                           train_df["lon_next"].to_numpy())
        m = float(np.mean(err))
        if m < best_mean:
            best_mean = m
            best_alpha = a
    return best_alpha, best_mean

# ============================================================
# 7) FEATURES
# ============================================================
def build_features(df_step, alpha):
    u_phys = df_step["uo"].to_numpy() + alpha * df_step["u10"].to_numpy()
    v_phys = df_step["vo"].to_numpy() + alpha * df_step["v10"].to_numpy()

    du = df_step["v_true_e"].to_numpy() - u_phys
    dv = df_step["v_true_n"].to_numpy() - v_phys

    doy = df_step["time"].dt.dayofyear.astype(float).to_numpy()
    sin_doy = np.sin(2*np.pi*doy/365.25)
    cos_doy = np.cos(2*np.pi*doy/365.25)

    speed_c = np.sqrt(df_step["uo"].to_numpy()**2 + df_step["vo"].to_numpy()**2)
    speed_w = np.sqrt(df_step["u10"].to_numpy()**2 + df_step["v10"].to_numpy()**2)

    lat = df_step["latitude"].to_numpy()
    lon = df_step["longitude"].to_numpy()

    X = np.column_stack([
        df_step["uo"].to_numpy(), df_step["vo"].to_numpy(),
        df_step["u10"].to_numpy(), df_step["v10"].to_numpy(),
        speed_c, speed_w,
        sin_doy, cos_doy,
        lat, lon,
        df_step["wind_missing"].to_numpy()
    ])
    y_res = np.column_stack([du, dv])
    return X, y_res, u_phys, v_phys

def build_pure_ml_targets(df_step):
    return np.column_stack([df_step["v_true_e"].to_numpy(), df_step["v_true_n"].to_numpy()])

# ============================================================
# 8) MODELS
# ============================================================
def make_extratrees(seed):
    base = ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed
    )
    return MultiOutputRegressor(base)

def make_ridge():
    return MultiOutputRegressor(Ridge(alpha=2.0, random_state=SEED))

# ============================================================
# 9) DYNAMICS
# ============================================================
def step_forward(lat, lon, u, v):
    lat2 = lat + (v*dt)/R*(180/np.pi)
    lon2 = lon + (u*dt)/(R*np.cos(np.radians(lat)))*(180/np.pi)
    return lat2, lon_wrap(lon2)

def rollout_endpoint_error(traj_df, start_idx, steps, alpha, mode, model_res=None, model_pure=None):
    lat = float(traj_df.loc[start_idx, "latitude"])
    lon = float(traj_df.loc[start_idx, "longitude"])

    for k in range(steps):
        r = traj_df.iloc[start_idx + k]
        uo = float(r["uo"]); vo = float(r["vo"])
        u10 = float(r["u10"]); v10 = float(r["v10"])
        wmiss = float(r["wind_missing"])

        u_phys = uo + alpha*u10
        v_phys = vo + alpha*v10

        if mode in ["hybrid", "linear_resid"]:
            t = pd.to_datetime(r["time"]).to_pydatetime()
            doy = int(t.timetuple().tm_yday)
            sin_doy = float(np.sin(2*np.pi*doy/365.25))
            cos_doy = float(np.cos(2*np.pi*doy/365.25))
            speed_c = float(np.sqrt(uo**2 + vo**2))
            speed_w = float(np.sqrt(u10**2 + v10**2))

            Xk = np.array([[uo, vo, u10, v10, speed_c, speed_w,
                            sin_doy, cos_doy, lat, lon, wmiss]])
            res = model_res.predict(Xk)[0]
            u = u_phys + res[0]
            v = v_phys + res[1]

        elif mode == "pure_ml":
            t = pd.to_datetime(r["time"]).to_pydatetime()
            doy = int(t.timetuple().tm_yday)
            sin_doy = float(np.sin(2*np.pi*doy/365.25))
            cos_doy = float(np.cos(2*np.pi*doy/365.25))
            speed_c = float(np.sqrt(uo**2 + vo**2))
            speed_w = float(np.sqrt(u10**2 + v10**2))

            Xk = np.array([[uo, vo, u10, v10, speed_c, speed_w,
                            sin_doy, cos_doy, lat, lon, wmiss]])
            uv = model_pure.predict(Xk)[0]
            u = uv[0]; v = uv[1]

        else:
            u = u_phys
            v = v_phys

        lat, lon = step_forward(lat, lon, u, v)

    lat_true = float(traj_df.loc[start_idx + steps, "latitude"])
    lon_true = float(traj_df.loc[start_idx + steps, "longitude"])
    return float(haversine_km(lat, lon, lat_true, lon_true))

# ============================================================
# 10) BACKTRACKING (reverse integration)
# ============================================================
def reverse_integrate_candidates(lat, lon, window, alpha, mode, model_res=None):
    lat = lat.copy()
    lon = lon.copy()
    n = len(lat)

    for _, r in window.iloc[::-1].iterrows():
        uo = float(r["uo"]); vo = float(r["vo"])
        u10 = float(r["u10"]); v10 = float(r["v10"])
        wmiss = float(r["wind_missing"])
        t = pd.to_datetime(r["time"]).to_pydatetime()

        u_phys = uo + alpha*u10
        v_phys = vo + alpha*v10

        if mode in ["hybrid", "linear_resid"]:
            doy = int(t.timetuple().tm_yday)
            sin_doy = float(np.sin(2*np.pi*doy/365.25))
            cos_doy = float(np.cos(2*np.pi*doy/365.25))
            speed_c = float(np.sqrt(uo**2 + vo**2))
            speed_w = float(np.sqrt(u10**2 + v10**2))

            Xk = np.column_stack([
                np.full(n, uo), np.full(n, vo),
                np.full(n, u10), np.full(n, v10),
                np.full(n, speed_c), np.full(n, speed_w),
                np.full(n, sin_doy), np.full(n, cos_doy),
                lat, lon,
                np.full(n, wmiss)
            ])
            res = model_res.predict(Xk)
            u = u_phys + res[:, 0]
            v = v_phys + res[:, 1]
        else:
            u = np.full(n, u_phys)
            v = np.full(n, v_phys)

        lat = lat - (v*dt)/R*(180/np.pi)
        lon = lon - (u*dt)/(R*np.cos(np.radians(lat)))*(180/np.pi)
        lon = lon_wrap(lon)

    return lat, lon

def summarize_survivors(surv_latlon, true_origin_latlon):
    if len(surv_latlon) == 0:
        return {"n": 0, "mean_origin_km": np.nan, "p90_origin_km": np.nan}
    d = haversine_km(
        surv_latlon[:, 0], surv_latlon[:, 1],
        true_origin_latlon[0], true_origin_latlon[1]
    )
    return {
        "n": int(len(surv_latlon)),
        "mean_origin_km": float(np.mean(d)),
        "p90_origin_km": float(np.quantile(d, 0.90))
    }

# ============================================================
# 11) GROUP CV (6h one-step)
# ============================================================
alphas = [0.0, 0.01, 0.02, 0.03, 0.04]
gkf = GroupKFold(n_splits=5)

cv_rows = []
fold = 0

for tr_idx, te_idx in gkf.split(step, groups=groups):
    fold += 1
    train = step.iloc[tr_idx].copy()
    test  = step.iloc[te_idx].copy()

    best_alpha, _ = tune_alpha(train, alphas)
    Xtr, ytr, _, _ = build_features(train, best_alpha)
    Xte, yte, u_phys_te, v_phys_te = build_features(test, best_alpha)

    model_hyb = make_extratrees(SEED + fold)
    model_lin = make_ridge()
    model_hyb.fit(Xtr, ytr)
    model_lin.fit(Xtr, ytr)

    ytr_pure = build_pure_ml_targets(train)
    model_pure = make_extratrees(SEED + 100 + fold)
    model_pure.fit(Xtr, ytr_pure)

    lat0 = test["latitude"].to_numpy()
    lon0 = test["longitude"].to_numpy()
    lat_true = test["lat_next"].to_numpy()
    lon_true = test["lon_next"].to_numpy()

    lat_p, lon_p = step_forward(lat0, lon0, u_phys_te, v_phys_te)
    err_phys = haversine_km(lat_p, lon_p, lat_true, lon_true)

    pred_res = model_hyb.predict(Xte)
    u_h = u_phys_te + pred_res[:, 0]
    v_h = v_phys_te + pred_res[:, 1]
    lat_h, lon_h = step_forward(lat0, lon0, u_h, v_h)
    err_hyb = haversine_km(lat_h, lon_h, lat_true, lon_true)

    pred_lin = model_lin.predict(Xte)
    u_l = u_phys_te + pred_lin[:, 0]
    v_l = v_phys_te + pred_lin[:, 1]
    lat_l, lon_l = step_forward(lat0, lon0, u_l, v_l)
    err_lin = haversine_km(lat_l, lon_l, lat_true, lon_true)

    pred_uv = model_pure.predict(Xte)
    u_pm = pred_uv[:, 0]
    v_pm = pred_uv[:, 1]
    lat_pm, lon_pm = step_forward(lat0, lon0, u_pm, v_pm)
    err_pure = haversine_km(lat_pm, lon_pm, lat_true, lon_true)

    cv_rows.append({
        "fold": fold,
        "best_alpha": best_alpha,
        "phys_mae_6h": float(np.mean(err_phys)),
        "hyb_mae_6h": float(np.mean(err_hyb)),
        "lin_mae_6h": float(np.mean(err_lin)),
        "pure_mae_6h": float(np.mean(err_pure)),
        "hyb_skill_6h": skill(float(np.mean(err_hyb)), float(np.mean(err_phys))),
        "lin_skill_6h": skill(float(np.mean(err_lin)), float(np.mean(err_phys))),
        "pure_skill_6h": skill(float(np.mean(err_pure)), float(np.mean(err_phys))),
    })

cv_df = pd.DataFrame(cv_rows)
print("\n=== 5-Fold Group CV (Unseen Drifters) — 6h Endpoint MAE (km) ===")
print(cv_df.to_string(index=False))
print("\nMean ± Std (over folds):")
print(cv_df.drop(columns=["fold"]).agg(["mean", "std"]).to_string())

# ============================================================
# 12) TRAIN FINAL MODELS ON ONE HOLDOUT SPLIT
# ============================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(step, groups=groups))
train = step.iloc[train_idx].copy()
test  = step.iloc[test_idx].copy()

best_alpha, _ = tune_alpha(train, alphas)
print("\nFinal split best alpha:", best_alpha)

Xtr, ytr, _, _ = build_features(train, best_alpha)
Xte, yte, u_phys_te, v_phys_te = build_features(test, best_alpha)

model_hyb = make_extratrees(SEED)
model_lin = make_ridge()
model_hyb.fit(Xtr, ytr)
model_lin.fit(Xtr, ytr)

ytr_pure = build_pure_ml_targets(train)
model_pure = make_extratrees(SEED + 999)
model_pure.fit(Xtr, ytr_pure)

# 6h distribution plot (holdout)
lat0 = test["latitude"].to_numpy()
lon0 = test["longitude"].to_numpy()
lat_true = test["lat_next"].to_numpy()
lon_true = test["lon_next"].to_numpy()

lat_p, lon_p = step_forward(lat0, lon0, u_phys_te, v_phys_te)
err_phys_6h = haversine_km(lat_p, lon_p, lat_true, lon_true)

pred_res = model_hyb.predict(Xte)
u_h = u_phys_te + pred_res[:, 0]
v_h = v_phys_te + pred_res[:, 1]
lat_h, lon_h = step_forward(lat0, lon0, u_h, v_h)
err_hyb_6h = haversine_km(lat_h, lon_h, lat_true, lon_true)

print("\n=== 6h One-step (Holdout unseen drifters) ===")
print("Physics:", summarize_err(err_phys_6h))
print("Hybrid :", summarize_err(err_hyb_6h), "Skill:", skill(summarize_err(err_hyb_6h)["mae_km"],
                                                           summarize_err(err_phys_6h)["mae_km"]))

print("\n[FIGURE A] 6h Error Histogram (Holdout)")
plt.figure(figsize=(7,5))
plt.hist(err_phys_6h, bins=35, alpha=0.6, label="Physics")
plt.hist(err_hyb_6h,  bins=35, alpha=0.6, label="Hybrid")
plt.xlabel("6-hour endpoint error (km)")
plt.ylabel("Frequency")
plt.title("6h Drift Error Distribution (Holdout Unseen Drifters)")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# ============================================================
# 13) MULTI-STEP FORWARD ROLLOUT + 5 MAIN IEEE FIGURES
# ============================================================
traj_map = {did: g.reset_index(drop=True) for did, g in df.groupby("drifter_id")}

def rollout_eval(mode, hours, n_samples=400):
    steps = hours // DT_HOURS
    errs = []

    drifters = list(traj_map.keys())
    attempts = 0
    max_attempts = n_samples * ROLLOUT_ATTEMPT_MULT

    while len(errs) < n_samples and attempts < max_attempts:
        attempts += 1
        did = np.random.choice(drifters)
        traj = traj_map[did]
        if len(traj) < steps + 2:
            continue
        start_idx = np.random.randint(0, len(traj) - steps - 1)

        e = rollout_endpoint_error(
            traj, start_idx, steps, best_alpha, mode,
            model_res=(model_hyb if mode == "hybrid" else model_lin),
            model_pure=model_pure
        )
        if np.isfinite(e):
            errs.append(e)

    return np.array(errs, dtype=float)

print("\n=== Forward Rollout Endpoint Errors (Holdout-style evaluation) ===")

rollout_store = {}
rows = []

for H in ROLLOUT_HOURS:  # <-- includes 48h
    if H % DT_HOURS != 0:
        continue

    e_phys = rollout_eval("physics", H, n_samples=ROLLOUT_SAMPLES_PER_H)
    e_hyb  = rollout_eval("hybrid",  H, n_samples=ROLLOUT_SAMPLES_PER_H)
    e_lin  = rollout_eval("linear_resid", H, n_samples=ROLLOUT_SAMPLES_PER_H)
    e_pure = rollout_eval("pure_ml", H, n_samples=ROLLOUT_SAMPLES_PER_H)

    rollout_store[H] = {"Physics": e_phys, "Hybrid": e_hyb, "Linear": e_lin, "PureML": e_pure}

    s_phys = summarize_err(e_phys)
    s_hyb  = summarize_err(e_hyb)
    s_lin  = summarize_err(e_lin)
    s_pure = summarize_err(e_pure)

    print(f"\nH={H}h")
    print("Physics:", s_phys)
    print("Hybrid :", s_hyb,  "Skill:", skill(s_hyb["mae_km"],  s_phys["mae_km"]))
    print("Linear :", s_lin,  "Skill:", skill(s_lin["mae_km"],  s_phys["mae_km"]))
    print("PureML :", s_pure, "Skill:", skill(s_pure["mae_km"], s_phys["mae_km"]))

    rows += [
        {"H_h": H, "Model": "Physics", **s_phys, "skill": np.nan},
        {"H_h": H, "Model": "Hybrid",  **s_hyb,  "skill": skill(s_hyb["mae_km"],  s_phys["mae_km"])},
        {"H_h": H, "Model": "Linear",  **s_lin,  "skill": skill(s_lin["mae_km"],  s_phys["mae_km"])},
        {"H_h": H, "Model": "PureML",  **s_pure, "skill": skill(s_pure["mae_km"], s_phys["mae_km"])},
    ]

rollout_df = pd.DataFrame(rows)
print("\n=== Forward Rollout Summary Table (Ready for Results Section) ===")
print(rollout_df.sort_values(["H_h","Model"]).to_string(index=False))

Hs = np.array(sorted(rollout_store.keys()), dtype=float)
models_all = ["Physics", "Linear", "PureML", "Hybrid"]

def get_series(metric, model):
    out = []
    for H in Hs:
        row = rollout_df[(rollout_df["H_h"] == int(H)) & (rollout_df["Model"] == model)]
        out.append(float(row.iloc[0][metric]))
    return np.array(out, dtype=float)

print("\n[FIGURE 1] RMSE vs Horizon")
plt.figure(figsize=(7.5,5.2))
for m in models_all:
    plt.plot(Hs, get_series("rmse_km", m), marker="o", label=m)
plt.xlabel("Horizon (hours)")
plt.ylabel("RMSE (km)")
plt.title("Endpoint Error vs Horizon (RMSE)")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

print("\n[FIGURE 2] Skill vs Horizon")
plt.figure(figsize=(7.5,5.2))
for m in ["Linear", "PureML", "Hybrid"]:
    plt.plot(Hs, get_series("skill", m), marker="o", label=m)
plt.xlabel("Horizon (hours)")
plt.ylabel("Skill (1 - MAE_model / MAE_physics)")
plt.title("Skill vs Horizon (Higher is better)")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

print("\n[FIGURE 3] ECDF at 48h (distribution)")
H_ecdf = 48 if 48 in rollout_store else int(Hs.max())
plt.figure(figsize=(7.5,5.2))
for m in ["Physics", "Linear", "PureML", "Hybrid"]:
    x, y = ecdf(rollout_store[H_ecdf][m])
    plt.plot(x, y, label=m)
plt.xlabel(f"Endpoint error at {H_ecdf}h (km)")
plt.ylabel("CDF  (P(error ≤ x))")
plt.title(f"Error Distribution (ECDF) at {H_ecdf}h")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

print("\n[FIGURE 4] Tail Risk (p95) vs Horizon")
plt.figure(figsize=(7.5,5.2))
for m in models_all:
    plt.plot(Hs, get_series("p95_km", m), marker="o", label=m)
plt.xlabel("Horizon (hours)")
plt.ylabel("p95 endpoint error (km)")
plt.title("Tail Risk vs Horizon (p95)")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

# ============================================================
# 14) BACKTRACKING + BOX PLOTS
# ============================================================
eligible = df.groupby("drifter_id").size()
eligible = eligible[eligible >= (max(HORIZONS_HOURS)//DT_HOURS + 2)].index.to_numpy()
print("\nEligible drifters for max horizon:", len(eligible))

def run_backtracking(horizon_hours, eps_km, n_trials):
    steps = horizon_hours // DT_HOURS
    out_rows = []

    for trial in range(n_trials):
        did = np.random.choice(eligible)
        traj = traj_map[did]
        if len(traj) < steps + 2:
            continue
        start_idx = np.random.randint(0, len(traj) - steps - 1)

        true_origin = traj.loc[start_idx, ["latitude","longitude"]].values.astype(float)
        obs_point   = traj.loc[start_idx + steps, ["latitude","longitude"]].values.astype(float)
        window      = traj.iloc[start_idx:start_idx + steps].copy()

        lat_scale = BOX_KM*1000 / meters_per_degree_lat()
        lon_scale = BOX_KM*1000 / meters_per_degree_lon(obs_point[0])

        cand_lat_end = obs_point[0] + np.random.uniform(-lat_scale, lat_scale, N_CANDIDATES)
        cand_lon_end = obs_point[1] + np.random.uniform(-lon_scale, lon_scale, N_CANDIDATES)

        lat_o_phys, lon_o_phys = reverse_integrate_candidates(
            cand_lat_end, cand_lon_end, window, best_alpha, mode="physics"
        )
        lat_o_hyb, lon_o_hyb = reverse_integrate_candidates(
            cand_lat_end, cand_lon_end, window, best_alpha, mode="hybrid", model_res=model_hyb
        )

        d_phys = haversine_km(lat_o_phys, lon_o_phys, true_origin[0], true_origin[1])
        d_hyb  = haversine_km(lat_o_hyb,  lon_o_hyb,  true_origin[0], true_origin[1])

        phys_surv = np.column_stack([lat_o_phys, lon_o_phys])[d_phys <= eps_km]
        hyb_surv  = np.column_stack([lat_o_hyb,  lon_o_hyb])[d_hyb  <= eps_km]

        ps = summarize_survivors(phys_surv, true_origin)
        hs = summarize_survivors(hyb_surv, true_origin)

        out_rows.append({
            "trial": trial,
            "drifter_id": did,
            "horizon_h": horizon_hours,
            "eps_km": eps_km,
            "phys_n": ps["n"],
            "hyb_n": hs["n"],
            "phys_mean_origin_km": ps["mean_origin_km"],
            "hyb_mean_origin_km": hs["mean_origin_km"],
            "phys_p90_origin_km": ps["p90_origin_km"],
            "hyb_p90_origin_km": hs["p90_origin_km"],
        })

    return pd.DataFrame(out_rows)

all_res = []
for H in HORIZONS_HOURS:
    for EPS in EPS_LIST_KM:
        print(f"Running backtracking: horizon={H}h, EPS={EPS}km ...")
        all_res.append(run_backtracking(H, EPS, N_TRIALS))

res = pd.concat(all_res, ignore_index=True)

print("\n=== Backtracking Summary (mean over trials) ===")
summary = (
    res.groupby(["horizon_h", "eps_km"], as_index=False)
       .agg(
           phys_n=("phys_n", "mean"),
           hyb_n=("hyb_n", "mean"),
           phys_mean_origin_km=("phys_mean_origin_km", "mean"),
           hyb_mean_origin_km=("hyb_mean_origin_km", "mean"),
           phys_p90_origin_km=("phys_p90_origin_km", "mean"),
           hyb_p90_origin_km=("hyb_p90_origin_km", "mean"),
       )
)
print(summary.to_string(index=False))

# ============================================================
# 15) ORIGIN CLOUD (FIGURE 5)
# ============================================================
print("\n[FIGURE 5] Origin Cloud (Backtracking) at 48h, EPS=10km")
DEMO_H = 48
DEMO_EPS = 10
demo_steps = DEMO_H // DT_HOURS

did = np.random.choice(eligible)
traj = traj_map[did]
start_idx = 2 if len(traj) > demo_steps + 3 else 0

true_origin = traj.loc[start_idx, ["latitude","longitude"]].values.astype(float)
obs_point   = traj.loc[start_idx + demo_steps, ["latitude","longitude"]].values.astype(float)
window      = traj.iloc[start_idx:start_idx + demo_steps].copy()

lat_scale = BOX_KM*1000 / meters_per_degree_lat()
lon_scale = BOX_KM*1000 / meters_per_degree_lon(obs_point[0])

cand_lat_end = obs_point[0] + np.random.uniform(-lat_scale, lat_scale, N_CANDIDATES)
cand_lon_end = obs_point[1] + np.random.uniform(-lon_scale, lon_scale, N_CANDIDATES)

lat_o_phys, lon_o_phys = reverse_integrate_candidates(
    cand_lat_end, cand_lon_end, window, best_alpha, mode="physics"
)
lat_o_hyb, lon_o_hyb = reverse_integrate_candidates(
    cand_lat_end, cand_lon_end, window, best_alpha, mode="hybrid", model_res=model_hyb
)

d_phys = haversine_km(lat_o_phys, lon_o_phys, true_origin[0], true_origin[1])
d_hyb  = haversine_km(lat_o_hyb,  lon_o_hyb,  true_origin[0], true_origin[1])

phys_surv = np.column_stack([lat_o_phys, lon_o_phys])[d_phys <= DEMO_EPS]
hyb_surv  = np.column_stack([lat_o_hyb,  lon_o_hyb])[d_hyb  <= DEMO_EPS]

plt.figure(figsize=(6.8,6.8))
if len(phys_surv) > 0:
    plt.scatter(phys_surv[:,1], phys_surv[:,0], s=10, alpha=0.35, label="Physics origins")
if len(hyb_surv) > 0:
    plt.scatter(hyb_surv[:,1], hyb_surv[:,0], s=10, alpha=0.35, label="Hybrid origins")
plt.scatter(true_origin[1], true_origin[0], c="red", s=180, marker="*", label="True origin")
plt.scatter(obs_point[1], obs_point[0], c="black", s=90, marker="x", label="Observed endpoint")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Origin Cloud (H={DEMO_H}h, EPS={DEMO_EPS}km)")
plt.legend()
plt.grid(alpha=0.25)
plt.show()

print("\nDONE ✅ (48h included in rollout + origin cloud + 5 main figures shown)")
