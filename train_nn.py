"""
Train one small neural network per scenario from the box-sampling results (CPU).  One SLURM array task = one scenario.

Usage:   python train_nn.py <task_id>          task_id 0,1,2 -> scenario 1,2,3   (or set SLURM_ARRAY_TASK_ID)
SLURM:   see train_nn.sbatch   (sbatch --array=0-2 train_nn.sbatch)  -- submit it AFTER sample_boxes.py has finished.

Input : <DATA_DIR>/boxes64_scenario<s>.jsonl   written by sample_boxes.py
        rows with status "optimal" or "timeout_lb" that carry the first-stage solution y are used
        (label = optimal objective, or BARON's last proven lower bound for timeouts); infeasible / failed boxes are dropped.
        NN input  = molN, molW of the scenario's three streams (6 inputs), output = label.
Output (written when training is done, in <OUT_DIR>):
        model_s<s>_<TAG>.onnx            trained network, raw inputs -> raw output (input scaling and output scaling folded into the
                                         first / last layer). Training size, epochs and train/validation error are printed in the job's .out file.
"""
import os, sys, json, copy, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch, torch.nn as nn, onnx, onnxruntime as ort
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================== SETTINGS (edit these) ==============================
DATA_DIR   = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/decomp_srun"      # where sample_boxes.py wrote boxes64_scenario<s>.jsonl
OUT_DIR    = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/decomp_srun"      # where the trained network / data / metrics are written
HIDDEN     = [4, 4]                 # hidden layer sizes
ACTIVATION = "sigmoid"              # "sigmoid" or "relu"   (must match --activation in certify_scenarios.py)
TAG        = "b64h4sig"             # model file tag -> model_s<s>_<TAG>.onnx   (e.g. "b64h4" for relu)
LR, BATCH_SIZE, EPOCHS, PATIENCE, VAL_FRAC, SEED = 1e-3, 16, 2000, 200, 0.2, 0
MIN_ROWS   = 10                     # stop if a scenario has fewer usable rows than this
SCENARIO_STREAMS = {1: [1, 2, 3], 2: [4, 5, 6], 3: [5, 7, 8]}     # NN inputs: molN, molW of each stream (stream 5 is shared)
# ====================================================================================

tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
if tid not in (0, 1, 2): sys.exit("usage: python train_nn.py <0|1|2>   (task id 0,1,2 -> scenario 1,2,3)")
SS = tid + 1
assert ACTIVATION in ("sigmoid", "relu")
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(1)         # CPU only

# ---------------- 1. data ----------------
path = os.path.join(DATA_DIR, f"boxes64_scenario{SS}.jsonl")
if not os.path.exists(path): sys.exit(f"data file not found: {path}  (has sample_boxes.py finished for scenario {SS}?)")
rows, seen = [], set()
for line in open(path):
    try: r = json.loads(line)
    except Exception: continue                                   # ignore a partially written last line
    if r.get("status") in ("optimal", "timeout_lb") and r.get("label") is not None and r.get("y"):
        if tuple(r["box"]) in seen: continue                     # (a resubmitted sampling job never duplicates, but be safe)
        seen.add(tuple(r["box"])); rows.append(r)
print(f"scenario {SS}: {len(rows)} usable rows in {path}", flush=True)
if len(rows) < MIN_ROWS: sys.exit(f"only {len(rows)} usable rows (< {MIN_ROWS}); not training")
y_names = list(rows[0]["y"].keys())                              # the 16 first-stage variables, in the instance's order
in_names = [f"mol{x}Stream{k}" for k in SCENARIO_STREAMS[SS] for x in "NW"]
assert all(n in y_names for n in in_names)
Y16 = np.array([[r["y"][n] for n in y_names] for r in rows])
X = np.array([[r["y"][n] for n in in_names] for r in rows], dtype=np.float32)
yv = np.array([r["label"] for r in rows], dtype=np.float32)
status = np.array([r["status"] for r in rows])
print(f"  inputs {in_names}\n  label range [{yv.min():.6g}, {yv.max():.6g}], std {yv.std():.4g} | optimal labels {(status == 'optimal').sum()}, lower-bound labels {(status != 'optimal').sum()}", flush=True)


# ---------------- 2. train (standardised inputs/outputs, Adam, early stopping) ----------------
Xtr, Xva, ytr, yva = train_test_split(X, yv, test_size=VAL_FRAC, random_state=SEED)
x_mean, x_std = Xtr.mean(0), Xtr.std(0); x_std[x_std < 1e-8] = 1.0
y_mean, y_std = float(ytr.mean()), float(ytr.std()) or 1.0
prep = lambda a, b: (torch.tensor((a - x_mean) / x_std), torch.tensor(((b - y_mean) / y_std)[:, None]))
Xtr_t, ytr_t = prep(Xtr, ytr); Xva_t, yva_t = prep(Xva, yva)
dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH_SIZE, shuffle=True)
act = nn.Sigmoid if ACTIVATION == "sigmoid" else nn.ReLU
layers, d = [], X.shape[1]
for h in HIDDEN: layers += [nn.Linear(d, h), act()]; d = h
layers.append(nn.Linear(d, 1)); net = nn.Sequential(*layers)
opt = torch.optim.Adam(net.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50)
lossf = nn.MSELoss(); best, best_state, bad = float("inf"), None, 0
t0 = time.time()
for ep in range(EPOCHS):
    net.train()
    for xb, yb in dl: opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    net.eval()
    with torch.no_grad(): va = lossf(net(Xva_t), yva_t).item()
    sched.step(va)
    if va < best - 1e-9: best, bad = va, 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
    else: bad += 1
    if bad >= PATIENCE: break
net.load_state_dict(best_state); net.eval()
print(f"  trained {ep + 1} epochs in {time.time() - t0:.0f}s (best standardised val MSE {best:.4g})", flush=True)

# ---------------- 3. export to ONNX (raw inputs -> raw output; scaling folded into first / last Linear) ----------------
ex = copy.deepcopy(net).eval(); lins = [l for l in ex if isinstance(l, nn.Linear)]
with torch.no_grad():
    first, last = lins[0], lins[-1]; sx, mx = torch.tensor(x_std), torch.tensor(x_mean)
    first.bias.copy_(first.bias - first.weight @ (mx / sx)); first.weight.copy_(first.weight / sx)
    last.weight.mul_(y_std); last.bias.copy_(last.bias * y_std + y_mean)
model_path = os.path.join(OUT_DIR, f"model_s{SS}_{TAG}.onnx")
_kw = dict(input_names=["input"], output_names=["output"], opset_version=13, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
try:      # newer PyTorch: force the legacy exporter so layers are plain Gemm nodes (what OMLT reads)
    torch.onnx.export(ex, torch.zeros(1, X.shape[1]), model_path, dynamo=False, **_kw)
except TypeError:   # older PyTorch (e.g. 2.1) has no `dynamo` argument; its default exporter is already the legacy one
    torch.onnx.export(ex, torch.zeros(1, X.shape[1]), model_path, **_kw)
onnx.checker.check_model(onnx.load(model_path))
# --- check the exported file ---
# (1) float64 evaluation of the ONNX network exactly as the solver will see it (its float32 weights, evaluated in double precision)
from onnx import numpy_helper
_g = onnx.load(model_path).graph; _W = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in _g.initializer}
def onnx_eval64(Xin):
    h = Xin.astype(np.float64)
    for nd in _g.node:
        if nd.op_type == "Gemm":
            tb = next((a.i for a in nd.attribute if a.name == "transB"), 0); Wm = _W[nd.input[1]]; h = h @ (Wm.T if tb else Wm) + _W[nd.input[2]]
        elif nd.op_type == "Sigmoid": h = 1.0 / (1.0 + np.exp(-h))
        elif nd.op_type == "Relu": h = np.maximum(h, 0.0)
        else: raise RuntimeError(f"unexpected ONNX node {nd.op_type}")
    return h.ravel()
_net64 = copy.deepcopy(net).double().eval()
with torch.no_grad(): ref64 = _net64(torch.tensor((X.astype(np.float64) - x_mean) / x_std)).numpy().ravel() * y_std + y_mean
scale = max(1.0, float(np.abs(yv).max()))
err = float(np.abs(onnx_eval64(X) - ref64).max())          # difference caused by storing the folded weights as float32
assert err <= 1e-3 * scale, f"ONNX export mismatch (float64 evaluation): {err}"
# (2) onnxruntime (float32 arithmetic) -- informational only; large input means / small stds make float32 evaluation of raw inputs inexact
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1       # explicit thread count: no CPU-affinity warnings
sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
err_ort = float(np.abs(sess.run(None, {"input": X})[0].ravel() - ref64).max())
print(f"  ONNX check: max |ONNX(float64) - torch| = {err:.3g} | onnxruntime float32 = {err_ort:.3g} (label scale {scale:.4g})", flush=True)
va_pred = onnx_eval64(Xva); tr_pred = onnx_eval64(Xtr)      # train/validation errors of the exported (float64-evaluated) network
m = dict(scenario=SS, tag=TAG, hidden=HIDDEN, activation=ACTIVATION, n_rows=len(yv), n_train=len(ytr), n_val=len(yva), epochs=ep + 1,
         train_rmse=float(np.sqrt(np.mean((tr_pred - ytr) ** 2))), val_rmse=float(np.sqrt(np.mean((va_pred - yva) ** 2))),
         val_mean_rel_err=float(np.mean(np.abs(va_pred - yva) / np.abs(yva))), label_std=float(yv.std()), onnx_max_err=err, model=model_path)
print(f"scenario {SS} DONE: train RMSE {m['train_rmse']:.4g} | val RMSE {m['val_rmse']:.4g} | val mean rel err {100 * m['val_mean_rel_err']:.2f}% | label std {m['label_std']:.4g}\n  network written to {model_path}", flush=True)
