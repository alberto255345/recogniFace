# scripts/train_liveness.py
# Treina um classificador (LR/SVC) usando as MESMAS features do pipeline de liveness.
# Usa preferencialmente a função do seu backend (app.liveness); se não existir, usa um fallback equivalente.

import argparse, os, json, glob
from pathlib import Path
import cv2
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# -------- tenta usar feature extractor do backend --------
FEAT_KEYS = [
    "blur", "saturation", "freq_high_ratio",
    "axis_anisotropy", "glare", "line_ratio", "penalty_sharp_small"
]
USE_APP = False
try:
    from app.liveness import extract_liveness_features as _extract
    def extract_features(bgr):
        d = _extract(bgr)  # deve retornar dict com as chaves acima
        return np.array([d.get(k, 0.0) for k in FEAT_KEYS], dtype="float32")
    USE_APP = True
except Exception:
    # -------- fallback: implementação aproximada --------
    def _safe_gray(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return g

    def _blur_score(bgr):
        g = _safe_gray(bgr)
        v = cv2.Laplacian(g, cv2.CV_64F).var()
        # normaliza ~[0..1] com saturação (hiperparâmetros simples)
        return float(np.tanh(v / 150.0))

    def _saturation(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[...,1].astype(np.float32)/255.0
        return float(s.mean())

    def _freq_high_ratio(bgr):
        g = _safe_gray(bgr).astype(np.float32) / 255.0
        G = np.fft.fftshift(np.fft.fft2(g))
        mag = np.log1p(np.abs(G))
        h, w = mag.shape
        cy, cx = h//2, w//2
        r = min(h,w)//6  # anel externo
        R = min(h,w)//2 - 1
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y-cy)**2 + (X-cx)**2)
        high = mag[(dist>=r) & (dist<=R)].mean() if np.any((dist>=r) & (dist<=R)) else 0.0
        low  = mag[(dist<r)].mean() if np.any(dist<r) else 1e-6
        return float(high / (high + low + 1e-6))

    def _axis_anisotropy(bgr):
        g = _safe_gray(bgr)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        sx = float(np.mean(np.abs(gx)) + 1e-6)
        sy = float(np.mean(np.abs(gy)) + 1e-6)
        an = abs(sx - sy) / (sx + sy)
        return float(np.clip(an, 0, 1))

    def _glare(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v = hsv[...,2].astype(np.float32)/255.0
        return float((v > 0.92).mean())

    def _line_ratio(bgr):
        g = _safe_gray(bgr)
        edges = cv2.Canny(g, 80, 180)
        return float(edges.mean())

    def _sharp_small(bgr):
        g = _safe_gray(bgr)
        # realça detalhe pequeno
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        sh = cv2.filter2D(g, -1, k).astype(np.float32) / 255.0
        m = float(np.mean(np.maximum(sh-0.6, 0.0)))
        return float(np.clip(m*2.0, 0, 1))

    def extract_features(bgr):
        f = [
            _blur_score(bgr),
            _saturation(bgr),
            _freq_high_ratio(bgr),
            _axis_anisotropy(bgr),
            _glare(bgr),
            _line_ratio(bgr),
            _sharp_small(bgr),
        ]
        return np.array(f, dtype="float32")

def load_images_from(dir_faces):
    exts = ('*.jpg','*.jpeg','*.png','*.bmp','*.webp')
    files = []
    for e in exts:
        files.extend(glob.glob(str(Path(dir_faces)/e)))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="Pasta com live/ e spoof/ (usamos subpasta faces/).")
    ap.add_argument("--out", required=True, help="Caminho do .joblib de saída.")
    ap.add_argument("--model", default="lr", choices=["lr","svc"], help="lr=LogisticRegression, svc=RBF SVC")
    ap.add_argument("--test_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_per_class", type=int, default=20, help="mínimo de amostras por classe para treinar")
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    live_dir  = train_dir/"live"/"faces"
    spoof_dir = train_dir/"spoof"/"faces"

    live_files  = load_images_from(live_dir)
    spoof_files = load_images_from(spoof_dir)

    print(f"[info] usando extractor: {'app.liveness' if USE_APP else 'fallback-integrado'}")
    print(f"[info] live:  {len(live_files)} imagens")
    print(f"[info] spoof: {len(spoof_files)} imagens")

    if len(live_files) < args.min_per_class or len(spoof_files) < args.min_per_class:
        raise SystemExit(f"Poucas amostras. Tenha >= {args.min_per_class} em cada classe (live/spoof).")

    X, y = [], []
    for lbl, files in [(1, live_files), (0, spoof_files)]:  # 1=live, 0=spoof
        for fp in files:
            im = cv2.imread(fp)
            if im is None:
                continue
            feat = extract_features(im)
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                continue
            X.append(feat); y.append(lbl)
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int64")

    print(f"[info] dataset final: X={X.shape}, y={y.shape}, FEAT_KEYS={FEAT_KEYS}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_split, random_state=args.seed, stratify=y)
    if args.model == "lr":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    else:
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    pipe.fit(Xtr, ytr)

    # avaliação
    yprob = pipe.predict_proba(Xte)[:,1]
    ypred = (yprob >= 0.5).astype(int)
    print("\n[report] holdout:")
    print(classification_report(yte, ypred, digits=4))
    try:
        auc = roc_auc_score(yte, yprob)
        print(f"[report] ROC-AUC: {auc:.4f}")
    except Exception:
        pass
    print("[report] confusion_matrix:\n", confusion_matrix(yte, ypred))

    # cross-val rápida
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"[cv] ROC-AUC (5-fold): mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # salva modelo + metadata
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    meta = {
        "feature_keys": FEAT_KEYS,
        "model": args.model,
        "train_dir": str(train_dir),
        "samples": {"live": len(live_files), "spoof": len(spoof_files)},
        "use_app_extractor": USE_APP,
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[ok] modelo salvo em: {out_path}")
    print(f"[ok] metadata: {out_path.with_suffix('.json')}")
    print("\nPara ativar no servidor:")
    print(f"  export LIVENESS_CLF_PATH={out_path}")
    print("  (reinicie o Uvicorn)")

if __name__ == "__main__":
    main()
