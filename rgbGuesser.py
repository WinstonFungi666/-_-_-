# B12504097 光電實驗week1__猜灰階值

#--------------------------------------------------------------------------------------------------------------#
GRID_PATH  = r"C:\Users\594ag\Desktop\光電實驗\LE\conbination.jpg"           # 整理成2*4陣列的五張圖片（上排 B,G,R,5；下排 4,3,2,1）
WHITE_PATH = r"C:\Users\594ag\Desktop\光電實驗\LE\whiteRef.png"              # 白色參考圖

OUT_CSV     = r"C:\Users\594ag\Desktop\光電實驗\LE\resulT.csv"               # 輸出的excel檔
OUT_PREVIEW = r"C:\Users\594ag\Desktop\光電實驗\LE\check.jpg"                # 取樣範圍檢查

gammaValue = 3.2650                                                         # from gammaGuesser
frac       = 0.20                                                           # 邊緣排除（避免取到邊界）
SAMPLES    = 1600                                                           # 中心取樣(其實900就夠了大概)
whiteFrac  = 0.95                                                           # 白圖中心裁切比例
#--------------------------------------------------------------------------------------------------------------#

import os, numpy as np, pandas as pd
from PIL import Image, ImageOps, ImageDraw

def srgbToLinear(u01):#this part is inspired by GPT5
    a = 0.055
    u01 = np.clip(u01, 0.0, 1.0)
    return np.where(u01 <= 0.04045, u01/12.92, ((u01 + a)/(1+a))**2.4)

def tileBbox(W,H, r,c, rows=2, cols=4, frac=0.2):#this part is powered by GPT5
    x0 = int(round(c * W/cols)); x1 = int(round((c+1) * W/cols))
    y0 = int(round(r * H/rows)); y1 = int(round((r+1) * H/rows))
    mx = int(round(frac * (x1-x0))); my = int(round(frac * (y1-y0)))
    return (x0+mx, y0+my, x1-mx, y1-my)

def regionGridMeans(img, bbox, samples=25):#this part is powered by GPT5
    x0, y0, x1, y1 = bbox
    h, w = y1-y0, x1-x0
    xs = np.linspace(0.15, 0.85, samples) * (w-1)
    ys = np.linspace(0.15, 0.85, samples) * (h-1)
    pts = [(int(round(x0+xx)), int(round(y0+yy))) for yy in ys for xx in xs]
    arr = np.asarray(img).astype(np.float64)
    vals = np.array([arr[py, px, :] for (px,py) in pts])
    return vals.mean(axis=0), vals.std(axis=0)

def centerCropBbox(W,H, frac=0.6):#this part is powered by GPT5
    w = int(round(W*frac))
    h = int(round(H*frac))
    x0 = (W-w)//2; y0 = (H-h)//2
    return x0,y0,x0+w,y0+h

def ECFP():
    gImg = Image.open(GRID_PATH)
    gImg = ImageOps.exif_transpose(gImg).convert("RGB")
    W,H = gImg.size

    rows, cols = 2,4
    means, stds, boxes = {}, {}, {}
    for r in range(rows):
        for c in range(cols):
            bbox = tileBbox(W,H, r,c, rows,cols, frac)
            m,s = regionGridMeans(gImg, bbox, samples=SAMPLES)
            means[(r,c)] = m; stds[(r,c)] = s; boxes[(r,c)] = bbox

    idx_B = (0,0); idx_G = (0,1); idx_R = (0,2); idx_5 = (0,3)
    idx_4 = (1,0); idx_3 = (1,1); idx_2 = (1,2); idx_1 = (1,3)

    cam_B = srgbToLinear(means[idx_B]/255.0)
    cam_G = srgbToLinear(means[idx_G]/255.0)
    cam_R = srgbToLinear(means[idx_R]/255.0)
    C = np.column_stack([cam_R, cam_G, cam_B])
    T = np.eye(3)
    try:
        M = T @ np.linalg.inv(C)
    except np.linalg.LinAlgError:
        M = T @ np.linalg.pinv(C)

    def toDispLinear(mean_u8):
        cam = srgbToLinear(mean_u8/255.0)
        return np.clip(M @ cam, 0.0, 4.0)

    wImg = Image.open(WHITE_PATH)
    wImg = ImageOps.exif_transpose(wImg).convert("RGB")
    wW,wH = wImg.size
    wx0,wy0,wx1,wy1 = centerCropBbox(wW,wH, frac=whiteFrac)
    wMean,_ = regionGridMeans(wImg, (wx0,wy0,wx1,wy1), samples=SAMPLES)
    wDisp = toDispLinear(wMean)
    whiteVec = np.clip(wDisp, 1e-6, None)

    inv_g = 1.0/float(gammaValue)
    def lin_vec_to_code(rgb_lin_rel):
        v = np.power(np.clip(rgb_lin_rel, 0.0, 2.0), inv_g) * 255.0
        return np.clip(np.round(v), 0, 255).astype(int)

#--------------------------------------------------------------------------------------------------------------# alig
    labels = ["5","4","3","2","1"]
    idx_map = {"5":idx_5,"4":idx_4,"3":idx_3,"2":idx_2,"1":idx_1}
    rows_out = []

    check_order = [("B_ref", idx_B), ("G_ref", idx_G), ("R_ref", idx_R)]
    for name, idx in check_order:
        m, s = means[idx], stds[idx]
        disp = toDispLinear(m)
        rel  = disp / whiteVec
        code = lin_vec_to_code(rel)
        rows_out.append({
            "label": name, "row": idx[0], "col": idx[1],
            "mean_R_u8": float(m[0]), "mean_G_u8": float(m[1]), "mean_B_u8": float(m[2]),
            "std_R_u8": float(s[0]),  "std_G_u8": float(s[1]),  "std_B_u8": float(s[2]),
            "disp_R_lin": float(disp[0]), "disp_G_lin": float(disp[1]), "disp_B_lin": float(disp[2]),
            "rel_R_lin": float(rel[0]),  "rel_G_lin": float(rel[1]),  "rel_B_lin": float(rel[2]),
            "code_R": int(code[0]), "code_G": int(code[1]), "code_B": int(code[2]),
        })

    for lab in labels:
        idx = idx_map[lab]
        m, s = means[idx], stds[idx]
        disp = toDispLinear(m)
        rel  = disp / whiteVec
        code = lin_vec_to_code(rel)
        rows_out.append({
            "label": lab, "row": idx[0], "col": idx[1],
            "mean_R_u8": float(m[0]), "mean_G_u8": float(m[1]), "mean_B_u8": float(m[2]),
            "std_R_u8": float(s[0]),  "std_G_u8": float(s[1]),  "std_B_u8": float(s[2]),
            "disp_R_lin": float(disp[0]), "disp_G_lin": float(disp[1]), "disp_B_lin": float(disp[2]),
            "rel_R_lin": float(rel[0]),  "rel_G_lin": float(rel[1]),  "rel_B_lin": float(rel[2]),
            "code_R": int(code[0]), "code_G": int(code[1]), "code_B": int(code[2]),
        })

#--------------------------------------------------------------------------------------------------------------# 輸出
    df = pd.DataFrame(rows_out)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    unk_rows = [r for r in rows_out if r["label"] in labels]
    unk_rows = sorted(unk_rows, key=lambda r: ["5","4","3","2","1"].index(r["label"]))
    rgb_list = [(r["code_R"], r["code_G"], r["code_B"]) for r in unk_rows]
    print("\nlabels = ['5','4','3','2','1']")
    print("RGB =", rgb_list)

    if OUT_PREVIEW:
        draw = ImageDraw.Draw(gImg)
        for (r,c),bbox in boxes.items():
            x0,y0,x1,y1 = bbox
            draw.rectangle([x0,y0,x1,y1], outline=(255,255,255), width=3)
        for idx,color in [(idx_B,(0,0,255)), (idx_G,(0,255,0)), (idx_R,(255,0,0))]:
            x0,y0,x1,y1 = boxes[idx]; draw.rectangle([x0,y0,x1,y1], outline=color, width=6)
        for idx in [idx_5, idx_4, idx_3, idx_2, idx_1]:
            x0,y0,x1,y1 = boxes[idx]; draw.rectangle([x0,y0,x1,y1], outline=(255,255,0), width=4)
        os.makedirs(os.path.dirname(OUT_PREVIEW), exist_ok=True)
        gImg.save(OUT_PREVIEW)
        print("Saved Preview:", OUT_PREVIEW)

    print("Saved excel file:", OUT_CSV)

if __name__ == "__main__":
    ECFP()
