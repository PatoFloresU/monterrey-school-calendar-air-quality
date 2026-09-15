"""
MIMA aplicado a la base horaria consolidada de este repositorio.

MIMA es el modelo de imputación por aprendizaje profundo (SAITS + CSDI en cascada)
desarrollado para la red SIMA por Emilio Gómez, Marcelo Sánchez, Jesús Medina, Luis
Montibeller y Santiago Pérez (Tecnológico de Monterrey), repositorio
`sntsemilio/MA2003B-Equipo-6-MIMA-v6`. Este script reproduce fielmente su protocolo
(arquitectura, ventana de 168 h, MinMaxScaler global, NaN->0 + máscara, enmascaramiento
artificial del 20%, Adam lr=1e-3, 50 épocas, lote 32, pérdida = MSE(SAITS) + MSE(ruido
CSDI), auditoría con huecos artificiales y verificación física PM2.5 <= PM10) sobre la
base horaria por zona construida en el notebook 00, con las adaptaciones siguientes,
todas documentadas:

  [A1] F = 5 variables (CO, NO2, O3, PM10, PM2.5): las cinco series del análisis.
  [A2] Validación química: correlación NO2 vs O3 (el modelo original usaba NOX vs O3).
  [A3] TRAIN_STRIDE = 24 h entre ventanas de entrenamiento (el original usa stride 1 en
       GPU). El filtro de validez (> 20% de celdas observadas por ventana) es idéntico.
  [A4] Entrenamiento en CPU con semillas fijas (reproducible).
  [A5] Imputación final con SAITS determinista; el refinamiento CSDI se entrena igual y
       su efecto se reporta como métrica, pero no se inyecta ruido al dataset de análisis.

Entradas:  ../base_horaria_zonas.csv.gz  (zona, fecha-hora, parámetro, valor)
Salidas:   ../datos_horarios_imputados_MIMA.csv.gz, mima_pato_model.pt,
           mima_auditoria.json, mima_imputacion_stats.json, mima_loss_hist.json

Uso:  cd data_processed/mima && python mima_pipeline.py                 (re-entrena, unas 2 h en CPU)
      cd data_processed/mima && python mima_pipeline.py --reuse-model   (carga mima_pato_model.pt y solo
                                                                        audita e imputa, unos minutos)

Datos: la base horaria por zona se construye en el notebook 00 a partir de los libros de SIMA, que no se
redistribuyen (SIMA los entregó para este estudio con la condición de no difundirlos). / Data: the zone
hourly base is built by notebook 00 from the SIMA workbooks, which are not redistributed (SIMA provided
them for this study on the condition that they are not disseminated).
"""
import os, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

torch.manual_seed(2026); np.random.seed(2026)
torch.set_num_threads(max(1, os.cpu_count() or 2))

AQUI = Path(__file__).resolve().parent          # data_processed/mima
DATA = AQUI.parent                               # data_processed
OUT = AQUI

CONFIG = {
    'seq_len': 168,
    'features': 5,
    'batch_size': 32,
    'hidden_size': 64,   # d_model del modelo original
    'diff_steps': 50,
    'epochs': 50,
    'train_stride': 24,  # [A3]
    'device': torch.device("cpu"),
}
FEATURES = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5']  # [A1]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ------------------------------------------------------------------ datos
def prepare_data():
    """Equivalente del procesador de datos del modelo original, partiendo de la base horaria por zona."""
    log("Cargando base_horaria_zonas.csv.gz ...")
    df = pd.read_csv(DATA / "base_horaria_zonas.csv.gz", parse_dates=["date"])
    stations = sorted(df["estacion"].unique())
    log(f"Zonas: {stations}")

    # scaler global ajustado sobre los valores crudos (ajuste por columna, como el original)
    wide_all = df.pivot_table(index=["date", "estacion"], columns="parametro",
                              values="valor", aggfunc="mean")[FEATURES]
    scaler = MinMaxScaler()
    scaler.fit(wide_all.values)  # ignora NaN en el ajuste (sklearn)

    t0 = df["date"].min().normalize()
    t1 = df["date"].max().normalize() + pd.Timedelta(hours=23)
    full_idx = pd.date_range(t0, t1, freq="h")
    data_list = []
    for st in stations:
        w = wide_all.xs(st, level="estacion").reindex(full_idx)
        vals = scaler.transform(w.values).astype(np.float32)
        data_list.append(vals)
        miss = np.isnan(vals).mean(axis=0)
        log(f"  {st}: {len(vals)} h | faltantes por var: " +
            ", ".join(f"{f}={m:.1%}" for f, m in zip(FEATURES, miss)))
    return stations, data_list, full_idx, scaler

def build_train_indices(data_list, seq_len, stride):
    """Filtro de validez idéntico al original (> 20% de celdas observadas por ventana)."""
    valid = []
    for i, arr in enumerate(data_list):
        obs = ~np.isnan(arr)
        n = len(arr); total_cells = seq_len * arr.shape[1]
        for s in range(0, n - seq_len + 1, stride):
            if obs[s:s+seq_len].sum() / total_cells > 0.2:
                valid.append((i, s))
    return valid

class LazyAirQualityDataset(Dataset):
    """Dataset por ventanas (misma lógica que el modelo original)."""
    def __init__(self, data_list, valid_indices, seq_len):
        self.data_list, self.valid_indices, self.seq_len = data_list, valid_indices, seq_len
    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        i, s = self.valid_indices[idx]
        window = self.data_list[i][s:s+self.seq_len]
        return {
            'observed_data': torch.from_numpy(np.nan_to_num(window, nan=0.0)).float(),
            'observed_mask': torch.from_numpy((~np.isnan(window)).astype(np.float32)),
        }

# ------------------------------------------------------------------ arquitectura (idéntica al modelo original)
class SAITS_Base(nn.Module):
    def __init__(self, num_features, seq_len, d_model=64, n_head=4):
        super().__init__()
        self.input_projection = nn.Linear(num_features * 2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_projection = nn.Linear(d_model, num_features)
    def forward(self, x, mask):
        x_combined = torch.cat([x, mask], dim=-1)
        x_emb = self.input_projection(x_combined)
        encoded = self.transformer(x_emb)
        imputed = self.output_projection(encoded)
        x_filled = x * mask + imputed * (1 - mask)
        return x_filled, imputed

class CSDI_Base(nn.Module):
    def __init__(self, num_features, seq_len, d_model=64):
        super().__init__()
        self.time_embedding = nn.Embedding(seq_len, d_model)
        self.feature_embedding = nn.Embedding(num_features, d_model)
        self.residual_layers = nn.Sequential(
            nn.Linear(num_features + d_model, 128), nn.ReLU(), nn.Linear(128, num_features))
        self.cond_projection = nn.Linear(num_features, d_model)
    def forward(self, x_noisy, t_step, saits_condition):
        cond_emb = self.cond_projection(saits_condition)
        net_input = torch.cat([x_noisy, cond_emb], dim=-1)
        return self.residual_layers(net_input)

class CascadeSOTA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.saits = SAITS_Base(config['features'], config['seq_len'], d_model=config['hidden_size'])
        self.csdi  = CSDI_Base(config['features'], config['seq_len'], d_model=config['hidden_size'])

# ------------------------------------------------------------------ entrenamiento (protocolo idéntico)
def train_scale_model(dataloader, config):
    model = CascadeSOTA(config).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    log(f"Entrenando {config['epochs']} épocas | {len(dataloader)} lotes/época ...")
    model.train()
    hist = []
    for epoch in range(config['epochs']):
        total_loss, t_start = 0.0, time.time()
        for batch in dataloader:
            x = batch['observed_data']; mask = batch['observed_mask']
            rand_mask = (torch.rand_like(x) > 0.2).float()
            target_mask = mask * rand_mask
            x_input = x * target_mask
            optimizer.zero_grad()
            noise = torch.randn_like(x)
            t = torch.randint(0, config['diff_steps'], (x.shape[0],))
            x_coarse, _ = model.saits(x_input, target_mask)
            pred_noise = model.csdi(noise, t, x_coarse)
            loss = F.mse_loss(x_coarse*mask, x*mask) + F.mse_loss(pred_noise, noise)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        avg = total_loss/len(dataloader); hist.append(avg)
        log(f"  Epoch {epoch+1}/{config['epochs']} | Loss {avg:.5f} | {time.time()-t_start:.0f}s")
    return model, hist

# ------------------------------------------------------------------ auditoría con huecos artificiales
def denorm(arr_norm, scaler):
    shp = arr_norm.shape
    return scaler.inverse_transform(arr_norm.reshape(-1, shp[-1])).reshape(shp)

def audit(model, dataloader, scaler, config, mask_ratio=0.2, max_batches=20, use_csdi=False, tag=""):
    model.eval()
    all_real, all_imp = [], []
    phys_flags, chem_pairs = [], []
    g = torch.Generator().manual_seed(777)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches: break
            x = batch['observed_data']; mask_orig = batch['observed_mask']
            rand_mask = (torch.rand(x.shape, generator=g) > mask_ratio).float()
            mask_input = mask_orig * rand_mask
            x_input = x * mask_input
            x_imp, _ = model.saits(x_input, mask_input)
            if use_csdi:  # refinamiento opcional (media de ensamble para no meter ruido)
                refs = []
                for k in range(10):
                    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(1000+k))
                    t = torch.zeros((x.shape[0],)).long()
                    pred_noise = model.csdi(noise, t, x_imp)
                    refs.append(x_imp + (noise - pred_noise) * 0.1)
                x_imp = torch.stack(refs).mean(0)
            eval_mask = (mask_orig == 1) & (rand_mask == 0)
            if eval_mask.sum() == 0: continue
            x_phys  = denorm(x.numpy(), scaler)
            xi_phys = denorm(x_imp.numpy(), scaler)
            m = eval_mask.numpy().astype(bool)
            all_real.append(x_phys[m]); all_imp.append(xi_phys[m])
            # física/química sobre las celdas imputadas del lote completo
            pm25 = xi_phys[:, :, FEATURES.index('PM2.5')]
            pm10 = xi_phys[:, :, FEATURES.index('PM10')]
            phys_flags.append((pm25 > pm10).mean())
            chem_pairs.append((xi_phys[:, :, FEATURES.index('NO2')].ravel(),
                               xi_phys[:, :, FEATURES.index('O3')].ravel()))
    y_true = np.concatenate(all_real); y_pred = np.concatenate(all_imp)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nz = y_true > 1.0
    mre = float(np.mean(np.abs(y_true[nz]-y_pred[nz]) / y_true[nz]) * 100)
    pear = pearsonr(y_true, y_pred)[0]
    chem = pearsonr(np.concatenate([c[0] for c in chem_pairs]),
                    np.concatenate([c[1] for c in chem_pairs]))[0]
    rep = {"tag": tag, "n_celdas_evaluadas": int(len(y_true)), "MAE": round(float(mae),4),
           "RMSE": round(float(rmse),4), "MRE_pct": round(mre,2), "R2": round(float(r2),4),
           "Pearson": round(float(pear),4), "viol_fisica_PM_pct": round(float(np.mean(phys_flags))*100,2),
           "quimica_NO2_O3_corr": round(float(chem),4)}
    log(f"AUDITORÍA {tag}: {rep}")
    return rep

# ------------------------------------------------------------------ imputación final (SAITS, determinista)
def impute_full(model, stations, data_list, full_idx, scaler, config):
    model.eval()
    L = config['seq_len']
    rows = []
    stats = {}
    with torch.no_grad():
        for si, st in enumerate(stations):
            arr = data_list[si]
            n = len(arr)
            filled = arr.copy()
            starts = list(range(0, n - L + 1, L))
            if starts[-1] != n - L: starts.append(n - L)  # cola
            for s in starts:
                window = arr[s:s+L]
                x = torch.from_numpy(np.nan_to_num(window, nan=0.0)).float().unsqueeze(0)
                m = torch.from_numpy((~np.isnan(window)).astype(np.float32)).unsqueeze(0)
                x_filled, _ = model.saits(x, m)
                out = x_filled.squeeze(0).numpy()
                sel = np.isnan(filled[s:s+L])
                seg = filled[s:s+L]; seg[sel] = out[sel]; filled[s:s+L] = seg
            phys = denorm(filled, scaler)
            neg = int((phys < 0).sum())
            phys = np.clip(phys, 0, None)  # sin concentraciones negativas
            was_nan = np.isnan(arr)
            stats[st] = {"celdas_imputadas": int(was_nan.sum()),
                         "pct_imputado": round(float(was_nan.mean())*100, 2),
                         "negativos_recortados": neg}
            dfw = pd.DataFrame(phys, index=full_idx, columns=FEATURES)
            dfl = dfw.reset_index(names="date").melt(id_vars="date", var_name="parametro", value_name="valor")
            dfl["estacion"] = st
            rows.append(dfl)
    out = pd.concat(rows, ignore_index=True)[["date","estacion","parametro","valor"]]
    out["valor"] = out["valor"].round(4)
    log(f"Imputación completa: {json.dumps(stats, ensure_ascii=False)}")
    return out, stats

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    import sys
    reusar_modelo = '--reuse-model' in sys.argv
    if not (DATA / 'base_horaria_zonas.csv.gz').exists():
        raise FileNotFoundError(
            "Falta data_processed/base_horaria_zonas.csv.gz: se construye en el notebook 00 a partir de los libros "
            "de SIMA, que no se redistribuyen. / Missing data_processed/base_horaria_zonas.csv.gz: notebook 00 builds "
            "it from the SIMA workbooks, which are not redistributed.")
    log("=== MIMA sobre la base horaria por zona: inicio ===")
    stations, data_list, full_idx, scaler = prepare_data()
    train_idx = build_train_indices(data_list, CONFIG['seq_len'], CONFIG['train_stride'])
    log(f"Ventanas de entrenamiento (stride {CONFIG['train_stride']}, filtro >20% obs): {len(train_idx)}")
    ds = LazyAirQualityDataset(data_list, train_idx, CONFIG['seq_len'])
    gen = torch.Generator().manual_seed(2026)
    dl = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True, generator=gen, num_workers=0)

    if reusar_modelo and (OUT / "mima_pato_model.pt").exists():
        model = CascadeSOTA(CONFIG).to(CONFIG['device'])
        model.load_state_dict(torch.load(OUT / "mima_pato_model.pt", map_location=CONFIG['device']))
        log("Modelo entrenado cargado de mima_pato_model.pt (sin re-entrenar). / Trained model loaded, no retraining.")
    else:
        model, hist = train_scale_model(dl, CONFIG)
        torch.save(model.state_dict(), OUT / "mima_pato_model.pt")
        json.dump(hist, open(OUT / "mima_loss_hist.json", "w"))
        log("Modelo guardado.")

    dl_eval = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True,
                         generator=torch.Generator().manual_seed(99), num_workers=0)
    rep_saits = audit(model, dl_eval, scaler, CONFIG, tag="SAITS")
    dl_eval2 = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True,
                          generator=torch.Generator().manual_seed(99), num_workers=0)
    rep_csdi = audit(model, dl_eval2, scaler, CONFIG, use_csdi=True, tag="SAITS+CSDI(ens10)")
    nota = ("La etapa CSDI, tal como está codificada en el modelo original, no altera las métricas "
            "(ensamble de 10 refinamientos); la imputación final del análisis es el SAITS determinista.")
    json.dump({"saits": rep_saits, "saits_csdi_ens10": rep_csdi, "nota": nota},
              open(OUT / "mima_auditoria.json", "w"), ensure_ascii=False, indent=2)

    hourly, stats = impute_full(model, stations, data_list, full_idx, scaler, CONFIG)
    hourly.to_csv(DATA / "datos_horarios_imputados_MIMA.csv.gz", index=False, compression="gzip")
    json.dump(stats, open(OUT / "mima_imputacion_stats.json", "w"), ensure_ascii=False, indent=2)
    log(f"Serie imputada guardada: {DATA / 'datos_horarios_imputados_MIMA.csv.gz'} ({len(hourly):,} filas)")
    log("=== MIMA: FIN ===")
