# B12504097 光電實驗week1__算CIE(x, y)

#--------------------------------------------------------------------------------------------------------------#
DEFAULT_SPD_CSV = r"C:\Users\594ag\Desktop\光電實驗\20250903-csv\OLEDM60deg8-204500.csv"        # 實驗數據
DEFAULT_CMF_CSV = r"C:\Users\594ag\Desktop\光電實驗\20250903-csv\CIE_xyz_1931_2deg.csv"         # 那個表
DEFAULT_START = 380.0
DEFAULT_END   = 780.0
DEFAULT_STEP  = 1.0
DEFAULT_OUT   = "result.csv"                                                                   # 輸出的excel檔
#--------------------------------------------------------------------------------------------------------------#

import argparse
import numpy as np
import pandas as pd

def loadCmf(cmf_csv):#this part is powered by GPT5
    df = pd.read_csv(cmf_csv)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    need = ["wavelength_nm", "x_bar", "y_bar", "z_bar"]
    for n in need:
        if n not in cols:
            raise ValueError(f"CMF CSV missing column: {n}")
    return df[["wavelength_nm", "x_bar", "y_bar", "z_bar"]].astype(float)

def loadSpdFromCsv(csv_path):#this part is powered by GPT5
    df = pd.read_csv(csv_path, encoding="cp950")
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    wav_col = None
    for c in ["wavelength_nm", "wavelength", "lambda", "lam", "wava(nm)", "wava", "nm"]:
        if c in cols: wav_col = c; break
    inten_col = None
    for c in ["i", "intensity", "radiance", "power", "spd", "[mw/m2/nm]", "mw/m2/nm"]:
        if c in cols: inten_col = c; break
    if wav_col is None or inten_col is None:
        if len(df.columns) == 2:
            wav_col, inten_col = df.columns[0], df.columns[1]
        else:
            raise ValueError("Somehow dead.")
    out = df[[wav_col, inten_col]].copy()
    out.columns = ["wavelength_nm", "I"]
    out["wavelength_nm"] = pd.to_numeric(out["wavelength_nm"], errors="coerce")
    out["I"] = pd.to_numeric(out["I"], errors="coerce")
    out = out.dropna()
    return out

def interpToGrid(x_src, y_src, grid):
    return np.interp(grid, x_src, y_src, left=0.0, right=0.0)

def xyzFromSpd(spd_df, cmf_df, start, end, step):
    grid = np.arange(start, end + 1e-9, step, dtype=float)
    Ig  = interpToGrid(spd_df["wavelength_nm"].to_numpy(), spd_df["I"].to_numpy(), grid)
    xg  = interpToGrid(cmf_df["wavelength_nm"].to_numpy(), cmf_df["x_bar"].to_numpy(), grid)
    yg  = interpToGrid(cmf_df["wavelength_nm"].to_numpy(), cmf_df["y_bar"].to_numpy(), grid)
    zg  = interpToGrid(cmf_df["wavelength_nm"].to_numpy(), cmf_df["z_bar"].to_numpy(), grid)
    X = np.trapz(Ig * xg, grid)
    Y = np.trapz(Ig * yg, grid)
    Z = np.trapz(Ig * zg, grid)
    return X, Y, Z

def xyFromXyz(X, Y, Z, eps=1e-12):
    denom = X + Y + Z
    if denom < eps:
        return float("nan"), float("nan")
    return X/denom, Y/denom

#--------------------------------------------------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser()#this part is powered by GPT5
    ap.add_argument("--spd_csv", default=DEFAULT_SPD_CSV, help="Spectrum CSV (two columns or with headers)")
    ap.add_argument("--cmf_csv", default=DEFAULT_CMF_CSV, help="CIE 1931 2° CMF CSV (wavelength_nm,x_bar,y_bar,z_bar)")
    ap.add_argument("--start", type=float, default=DEFAULT_START, help="Start wavelength (nm)")
    ap.add_argument("--end",   type=float, default=DEFAULT_END,   help="End wavelength (nm)")
    ap.add_argument("--step",  type=float, default=DEFAULT_STEP,  help="Sampling step (nm)")
    ap.add_argument("--out",   default=DEFAULT_OUT, help="Output CSV")
    args = ap.parse_args()

    cmf = loadCmf(args.cmf_csv)
    spd = loadSpdFromCsv(args.spd_csv)

    wl_min = max(spd["wavelength_nm"].min(), cmf["wavelength_nm"].min())
    wl_max = min(spd["wavelength_nm"].max(), cmf["wavelength_nm"].max())
    start = max(args.start, wl_min)
    end = min(args.end,   wl_max)

    X, Y, Z = xyzFromSpd(spd, cmf, start, end, args.step)
    x, y = xyFromXyz(X, Y, Z)

    out = pd.DataFrame([{
        "X": X, "Y": Y, "Z": Z, "x": x, "y": y,
        "wl_start_nm": start, "wl_end_nm": end, "wl_step_nm": args.step,
        "spd_path": args.spd_csv, "cmf_path": args.cmf_csv
    }])
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
