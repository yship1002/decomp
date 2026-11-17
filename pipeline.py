"""
Whole pipeline for ONE scenario of EDUnits_nocp in one file:   sample boxes (BARON)  ->  train (4,4) NN (CPU)  ->  certify NN (BARON).
One SLURM array task = one scenario; the three scenarios run independently.

Usage:   python pipeline.py <task_id>        task_id 0,1,2 -> scenario 1,2,3   (or set SLURM_ARRAY_TASK_ID)
SLURM:   sbatch pipeline.sbatch              (array 0-2)

Files written to OUT_DIR (<instance> = EDUnits_nocp, <s> = scenario 1,2,3):
  <instance>_<s>.jsonl     stage 1: one line per finished box (resumable: a re-run skips boxes already in the file)
  <instance>_<s>.onnx      stage 2: trained network, float64, raw inputs -> raw output
  <instance>_<s>.json      stage 3: delta (proven lower bound of min f_s - NN_s), incumbent, gaps, termination, box, sanity check
Stages can be switched off with RUN_SAMPLE / RUN_TRAIN / RUN_CERTIFY (e.g. to re-certify with existing files).

delta_s is the solver's last PROVEN lower bound, so for every feasible point inside the NN's input box   NN_s(y) + delta_s <= f_s(y,x).
"""
import os, sys, json, time, copy, itertools, importlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import onnx
from onnx import numpy_helper
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition as TC
from omlt import OmltBlock
from omlt.io.onnx import load_onnx_neural_network
from omlt.neuralnet import ReluBigMFormulation, FullSpaceSmoothNNFormulation
import math
import omlt.neuralnet.activations.smooth as _sm, omlt.neuralnet.nn_formulation as _nf
_sig_sym = _sm.sigmoid_activation_function
def _sig_safe(x):      # OMLT's numeric bound propagation overflows exp() on wide boxes with large weights; clamp for plain numbers only
    if isinstance(x, (int, float, np.floating)): return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, float(x)))))
    return _sig_sym(x)
_sm.sigmoid_activation_function = _sig_safe; _nf.sigmoid_activation_function = _sig_safe

# ============================== SETTINGS (edit these) ==============================
PROBLEM_MODULE = "NSPLIB.src.instances.EDUnits_nocp.EDUnits_nocp"
OUT_DIR   = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/decomp_srun"       # all files are written here
BARON_EXE = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/privatemodule/baron/baron"            # None -> found on PATH
BARON_LIC = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/privatemodule/baron/baronlice.txt"    # None -> default
RUN_SAMPLE, RUN_TRAIN, RUN_CERTIFY = True, True, True

# --- TIME LIMITS (seconds) ---
SAMPLE_TIME = 60                    # stage 1: BARON limit for EACH box solve (worst case number of boxes x SAMPLE_TIME in total)
TRAIN_TIME  = None                  # stage 2: wall-clock limit for training (None = no limit; training stops earlier by EPOCHS / PATIENCE)
CERT_TIME   = 600                   # stage 3: BARON limit for the certification solve
# keep the #SBATCH -t in pipeline.sbatch above  (number of boxes) x SAMPLE_TIME + TRAIN_TIME + CERT_TIME  plus a few minutes for model builds

# --- stage 1: sampling ---
SAMPLE_OPTIONS = {"MaxTime": SAMPLE_TIME, "EpsR": 1e-2, "EpsA": 1e-6, "TDo": 0, "AbsConFeasTol": 1e-2}     # BARON limit (s) and gap per box
# the 6 free first-stage coordinates and the interval each spans; each interval is split into BOXES_PER_DIM equal pieces
RANGES = {"N6": (1e-5, 10.0), "W6": (1e-5, 500.0), "N8": (1e-5, 10.0), "W8": (1e-5, 500.0),
          "N2": (1e-5, 0.0635099), "W2": (117.907, 123.446)}      # N2, W2: range allowed by the stage-1 constraints
COORD_VARS = {"N6": "molNStream6", "W6": "molWStream6", "N8": "molNStream8", "W8": "molWStream8", "N2": "molNStream2", "W2": "molWStream2"}
BOXES_PER_DIM = 4                   # number of equal pieces per free dimension: one int for all six, or a dict such as {"N6": 3, "W6": 2, "N8": 3, "W8": 2, "N2": 1, "W2": 1}
                                    # total boxes = product over the six dimensions (2 each -> 2^6 = 64; 3 each -> 729). Changing it needs a fresh <instance>_<s>.jsonl.
MAX_BOXES = None                                                  # e.g. 4 for a quick test; None = all boxes

# --- stage 2: training ---
HIDDEN     = [4, 4]
ACTIVATION = "sigmoid"              # "sigmoid" (OMLT full-space NLP) or "relu" (OMLT big-M MILP)
TAG        = "b64h4sig"
LR, BATCH_SIZE, EPOCHS, PATIENCE, VAL_FRAC, SEED = 1e-3, 16, 2000, 200, 0.2, 0
MIN_ROWS   = 10                     # stop if a scenario has fewer usable rows than this

# --- stage 3: certification ---
CERT_SOLVER = "baron"               # "baron" or "gurobi"
CERT_GAP    = 1e-2                  # relative gap target
CERT_BOX    = "original"            # y over which f_s - NN_s is minimised: "original" = the instance's ORIGINAL variable bounds (training data ignored),
                                    #                                       "training" = only the box spanned by the training inputs
TEE         = False                 # True: show the solver's live log

SCENARIO_STREAMS = {1: [1, 2, 3], 2: [4, 5, 6], 3: [5, 7, 8]}     # NN inputs: molN, molW of each stream (stream 5 is shared)
# ====================================================================================

tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
if tid not in (0, 1, 2): sys.exit("usage: python pipeline.py <0|1|2>   (task id 0,1,2 -> scenario 1,2,3)")
SS = tid + 1
assert ACTIVATION in ("sigmoid", "relu") and CERT_SOLVER in ("baron", "gurobi") and CERT_BOX in ("original", "training")
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)                          # NSPLIB writes log files relative to the repo root
OUT_DIR = OUT_DIR if os.path.isabs(OUT_DIR) else os.path.join(REPO, OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
INSTANCE   = PROBLEM_MODULE.split(".")[-1]                        # "EDUnits_nocp"
DATA_FILE  = os.path.join(OUT_DIR, f"{INSTANCE}_{SS}.jsonl")      # stage 1 output: sampled boxes
MODEL_FILE = os.path.join(OUT_DIR, f"{INSTANCE}_{SS}.onnx")       # stage 2 output: trained network
CERT_FILE  = os.path.join(OUT_DIR, f"{INSTANCE}_{SS}.json")       # stage 3 output: certification result
in_names = [f"mol{x}Stream{k}" for k in SCENARIO_STREAMS[SS] for x in "NW"]
_names = list(RANGES)
N_BOXES = int(np.prod([int(BOXES_PER_DIM[n]) if isinstance(BOXES_PER_DIM, dict) else int(BOXES_PER_DIM) for n in _names]))
if MAX_BOXES: N_BOXES = min(N_BOXES, MAX_BOXES)
T_START = time.time()
print(f"scenario {SS}: time limits -> sampling {SAMPLE_TIME}s per box (max {N_BOXES * SAMPLE_TIME / 60:.0f} min for {N_BOXES} boxes), training {TRAIN_TIME or 'no limit'}, certification {CERT_TIME}s", flush=True)
def num(v): return None if v is None or not np.isfinite(v) else float(v)


def build_model(with_sample_obj=False):
    """Fresh Pyomo model of the instance with only scenario SS (+ stage-1 constraints) active and the original objective off."""
    sm = importlib.import_module(PROBLEM_MODULE).const_model()
    m = sm.pyomo_model; scenarios = list(sm.scenarios); assert SS in scenarios
    prefix = lambda s: "g_0_scenario_" if s == 0 else f"g_{s}_"
    all_cons = list(m.component_objects(pyo.Constraint, active=True))
    SC = {s: [c for c in all_cons if c.name.startswith(prefix(s))] for s in scenarios}
    SC1 = [c for c in all_cons if c.name.startswith("g_0_") and not c.name.startswith("g_0_scenario_")]     # stage-1 constraints stay active
    assert sum(len(v) for v in SC.values()) + len(SC1) == len(all_cons), "constraint grouping mismatch"
    m.obj.deactivate()
    for t in scenarios:
        for c in SC[t]: c.activate() if t == SS else c.deactivate()
    return sm, m


def baron_solver():
    return pyo.SolverFactory("baron", executable=BARON_EXE) if BARON_EXE else pyo.SolverFactory("baron")


# ======================================== STAGE 1: sampling ========================================
def stage_sample():
    sm, m = build_model(); all_y = list(sm.y_set)
    m.f_obj = pyo.Objective(expr=sm.obj_expr[SS], sense=pyo.minimize)
    opts = dict(SAMPLE_OPTIONS)
    if BARON_LIC: opts["LicName"] = BARON_LIC
    baron = baron_solver()
    names = list(RANGES)
    nb = [int(BOXES_PER_DIM[n]) if isinstance(BOXES_PER_DIM, dict) else int(BOXES_PER_DIM) for n in names]; assert all(k >= 1 for k in nb)
    edges = [np.linspace(RANGES[n][0], RANGES[n][1], k + 1) for n, k in zip(names, nb)]       # piece j of dimension d = (edges[d][j], edges[d][j+1])
    boxes = list(itertools.product(*[range(k) for k in nb]))
    if MAX_BOXES: boxes = boxes[:MAX_BOXES]
    done = set()
    if os.path.exists(DATA_FILE):                                     # resume: skip boxes already written
        for line in open(DATA_FILE):
            try: rec0 = json.loads(line)
            except Exception: continue
            ix = tuple(rec0["box"])
            ok_grid = len(ix) == len(names) and all(0 <= ix[d] < nb[d] and np.isclose(rec0["lo"][d], edges[d][ix[d]], rtol=1e-9, atol=1e-12)
                                                    and np.isclose(rec0["hi"][d], edges[d][ix[d] + 1], rtol=1e-9, atol=1e-12) for d in range(len(names)))
            if not ok_grid: sys.exit(f"{DATA_FILE} was written with a different box grid than BOXES_PER_DIM={BOXES_PER_DIM}; move or delete it and run again")
            done.add(ix)
    print(f"[sample] scenario {SS}: {len(boxes)} boxes, {len(done)} already in {DATA_FILE}", flush=True)
    t_all = time.time(); counts = {}
    for k, idx in enumerate(boxes):
        if idx in done: continue
        L = [edges[d][idx[d]] for d in range(6)]; U = [edges[d][idx[d] + 1] for d in range(6)]
        for d, n in enumerate(names):
            v = m.y[COORD_VARS[n]]; v.setlb(L[d]); v.setub(U[d]); v.set_value(0.5 * (L[d] + U[d]))
        status, label, ltype, ub, yv, xv = "error", None, None, None, None, None
        t0 = time.time()
        try:
            r = baron.solve(m, tee=False, load_solutions=False, options=opts)
            tc = r.solver.termination_condition
            snap = lambda: ({n: float(m.y[n].value) for n in all_y},
                            {v.name: float(v.value) for v in m.component_data_objects(pyo.Var) if v.value is not None and not v.name.startswith("y[")})
            if tc in (TC.optimal, TC.globallyOptimal, TC.locallyOptimal):
                m.solutions.load_from(r); status, ltype = "optimal", "optimal"; label = float(pyo.value(m.f_obj)); yv, xv = snap()
            elif tc in (TC.infeasible, TC.invalidProblem):
                status = "infeasible"
            else:
                lb = r.problem.lower_bound
                if lb is not None and np.isfinite(lb) and abs(lb) < 1e20:
                    status, ltype, label = "timeout_lb", "lower_bound", float(lb)       # label = last proven lower bound
                    ubv = r.problem.upper_bound; ub = num(ubv) if ubv is not None and abs(ubv) < 1e20 else None
                    try: m.solutions.load_from(r); yv, xv = snap()
                    except Exception: pass
                else: status = "timeout_nolb"
            if ub is None and status == "optimal": ub = label
        except Exception as e:
            status = "error:" + type(e).__name__
        rec = dict(scenario=SS, box=list(idx), lo=[float(a) for a in L], hi=[float(a) for a in U], status=status, label=num(label),
                   label_type=ltype, upper_bound=num(ub), seconds=round(time.time() - t0, 2), y=yv, x=xv)
        with open(DATA_FILE, "a") as fh:                              # write each finished box immediately
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
        counts[status] = counts.get(status, 0) + 1
        print(f"[sample {k + 1}/{len(boxes)}] box {idx}: {status}" + (f" label {label:.6g}" if label is not None else "") + f" ({rec['seconds']}s)", flush=True)
    print(f"[sample] scenario {SS} DONE in {time.time() - t_all:.0f}s: {counts}", flush=True)


def read_rows():
    """Usable boxes: optimal or timeout_lb with a stored first-stage solution (same filter for training and certification)."""
    if not os.path.exists(DATA_FILE): sys.exit(f"data file not found: {DATA_FILE}")
    rows, seen = [], set()
    for line in open(DATA_FILE):
        try: r = json.loads(line)
        except Exception: continue                                    # partially written last line
        if r.get("status") in ("optimal", "timeout_lb") and r.get("label") is not None and r.get("y") and tuple(r["box"]) not in seen:
            seen.add(tuple(r["box"])); rows.append(r)
    return rows


def eval64(graph, Xin):
    """Exported network evaluated in float64 straight from the ONNX weights (what the solver sees)."""
    W = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in graph.initializer}
    h = Xin.astype(np.float64)
    for nd in graph.node:
        if nd.op_type == "Gemm":
            tb = next((a.i for a in nd.attribute if a.name == "transB"), 0); Wm = W[nd.input[1]]; h = h @ (Wm.T if tb else Wm) + W[nd.input[2]]
        elif nd.op_type == "Sigmoid": h = 1.0 / (1.0 + np.exp(-h))
        elif nd.op_type == "Relu": h = np.maximum(h, 0.0)
        else: raise RuntimeError(f"unexpected ONNX node {nd.op_type}")
    return h.ravel()


# ======================================== STAGE 2: training ========================================
def stage_train():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(1)            # CPU only
    rows = read_rows()
    print(f"[train] scenario {SS}: {len(rows)} usable rows in {DATA_FILE}", flush=True)
    if len(rows) < MIN_ROWS: sys.exit(f"only {len(rows)} usable rows (< {MIN_ROWS}); not training")
    X = np.array([[r["y"][n] for n in in_names] for r in rows], dtype=np.float32)
    yv = np.array([r["label"] for r in rows], dtype=np.float32); status = np.array([r["status"] for r in rows])
    print(f"  inputs {in_names}\n  label range [{yv.min():.6g}, {yv.max():.6g}], std {yv.std():.4g} | optimal labels {(status == 'optimal').sum()}, "
          f"lower-bound labels {(status != 'optimal').sum()}", flush=True)
    perm = np.random.RandomState(SEED).permutation(len(yv)); nva = max(1, int(np.ceil(VAL_FRAC * len(yv))))
    va_i, tr_i = perm[:nva], perm[nva:]
    Xtr, Xva, ytr, yva = X[tr_i], X[va_i], yv[tr_i], yv[va_i]
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
        if bad >= PATIENCE or (TRAIN_TIME and time.time() - t0 > TRAIN_TIME): break
    net.load_state_dict(best_state); net.eval()
    print(f"  trained {ep + 1} epochs in {time.time() - t0:.0f}s (best standardised val MSE {best:.4g})", flush=True)
    # export: scaling folded into first / last Linear, everything in float64 (float32 rounding of the folded weights shifts outputs by hundreds)
    ex = copy.deepcopy(net).double().eval(); lins = [l for l in ex if isinstance(l, nn.Linear)]
    with torch.no_grad():
        first, last = lins[0], lins[-1]; sx, mx = torch.tensor(x_std, dtype=torch.float64), torch.tensor(x_mean, dtype=torch.float64)
        first.bias.copy_(first.bias - first.weight @ (mx / sx)); first.weight.copy_(first.weight / sx)
        last.weight.mul_(y_std); last.bias.copy_(last.bias * y_std + y_mean)
    kw = dict(input_names=["input"], output_names=["output"], opset_version=13, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    dummy = torch.zeros(1, X.shape[1], dtype=torch.float64)
    try: torch.onnx.export(ex, dummy, MODEL_FILE, dynamo=False, **kw)       # newer PyTorch: legacy exporter -> plain Gemm nodes (what OMLT reads)
    except TypeError: torch.onnx.export(ex, dummy, MODEL_FILE, **kw)        # older PyTorch has no `dynamo` argument
    onnx.checker.check_model(onnx.load(MODEL_FILE))
    g = onnx.load(MODEL_FILE).graph
    net64 = copy.deepcopy(net).double().eval()
    with torch.no_grad(): ref64 = net64(torch.tensor((X.astype(np.float64) - x_mean) / x_std)).numpy().ravel() * y_std + y_mean
    scale = max(1.0, float(np.abs(yv).max())); err = float(np.abs(eval64(g, X) - ref64).max())
    assert err <= 1e-3 * scale, f"ONNX export mismatch (float64 evaluation): {err}"
    try:                                                                    # onnxruntime check is informational only
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        sess = ort.InferenceSession(MODEL_FILE, sess_options=so, providers=["CPUExecutionProvider"])
        err_ort = f"{float(np.abs(sess.run(None, {'input': X.astype(np.float64)})[0].ravel() - ref64).max()):.3g}"
    except Exception as e: err_ort = f"skipped ({type(e).__name__})"
    print(f"  ONNX check: max |ONNX(float64) - torch| = {err:.3g} | onnxruntime = {err_ort} (label scale {scale:.4g})", flush=True)
    tr_pred, va_pred = eval64(g, Xtr), eval64(g, Xva)
    print(f"[train] scenario {SS} DONE: train RMSE {np.sqrt(np.mean((tr_pred - ytr) ** 2)):.4g} | val RMSE {np.sqrt(np.mean((va_pred - yva) ** 2)):.4g} | "
          f"val mean rel err {100 * np.mean(np.abs(va_pred - yva) / np.abs(yva)):.2f}% | label std {yv.std():.4g}\n  network written to {MODEL_FILE}", flush=True)


# ======================================== STAGE 3: certification ========================================
def stage_certify():
    if not os.path.exists(MODEL_FILE): sys.exit(f"file not found: {MODEL_FILE}")
    dt = {t.data_type for t in onnx.load(MODEL_FILE).graph.initializer}
    if dt != {onnx.TensorProto.DOUBLE}: sys.exit(f"{MODEL_FILE} has weights of ONNX data type(s) {sorted(dt)}, expected float64 (11). Retrain.")
    rows = read_rows()
    X = np.array([[r["y"][n] for n in in_names] for r in rows]); yv = np.array([r["label"] for r in rows]); st = np.array([r["status"] for r in rows])
    lo, hi = X.min(0), X.max(0); pad = 1e-9 * np.maximum(np.abs(hi), 1.0); lo, hi = lo - pad, hi + pad          # NN input box = range of the training inputs
    wmax = max(float(np.abs(numpy_helper.to_array(t)).max()) for t in onnx.load(MODEL_FILE).graph.initializer)
    print(f"[certify] scenario {SS}: network weights float64, largest |weight or bias| = {wmax:.4g}", flush=True)
    sm, m = build_model()                                              # fresh model (sampling changed bounds/values of the first-stage variables)
    tr_lo, tr_hi = lo.copy(), hi.copy()                                # box spanned by the training inputs
    if CERT_BOX == "original": lo = np.array([m.y[n].bounds[0] for n in in_names], dtype=float); hi = np.array([m.y[n].bounds[1] for n in in_names], dtype=float)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), "network inputs need finite bounds"
    print(f"[certify] scenario {SS}: {len(rows)} training rows span " + ", ".join(f"{n}[{a:.6g},{b:.6g}]" for n, a, b in zip(in_names, tr_lo, tr_hi)), flush=True)
    print(f"[certify] scenario {SS}: y minimised over the {CERT_BOX.upper()} box " + ", ".join(f"{n}[{a:.6g},{b:.6g}]" for n, a, b in zip(in_names, lo, hi)), flush=True)
    net = load_onnx_neural_network(onnx.load(MODEL_FILE), input_bounds={i: (float(lo[i]), float(hi[i])) for i in range(len(in_names))})
    form = ReluBigMFormulation(net) if ACTIVATION == "relu" else FullSpaceSmoothNNFormulation(net)
    m.nn = OmltBlock(); m.nn.build_formulation(form)
    m.link = pyo.ConstraintList()
    for i, n in enumerate(in_names): m.link.add(m.nn.inputs[i] == m.y[n])        # NN input i == first-stage variable
    m.cert_obj = pyo.Objective(expr=sm.obj_expr[SS] - m.nn.outputs[0], sense=pyo.minimize)          # f_s - NN_s
    print(f"--- certification scenario {SS}: min f_s - NN_s with {CERT_SOLVER} (limit {CERT_TIME}s, gap {CERT_GAP:g}, {ACTIVATION} net {MODEL_FILE}) ---", flush=True)
    t0 = time.time()
    if CERT_SOLVER == "baron":
        opts = {"MaxTime": CERT_TIME, "EpsR": CERT_GAP, "EpsA": 1e-6, "TDo": 0, "AbsConFeasTol": 1e-2}
        if BARON_LIC: opts["LicName"] = BARON_LIC
        res = baron_solver().solve(m, tee=TEE, load_solutions=False, options=opts)
        tc = str(res.solver.termination_condition).split(".")[-1]; lb = float(res.problem.lower_bound)
        ub = res.problem.upper_bound; ub = float(ub) if ub is not None and abs(ub) < 1e20 else float("nan")
    else:
        from pyomo.contrib.solver.solvers.gurobi_direct_minlp import GurobiDirectMINLP
        res = GurobiDirectMINLP().solve(m, load_solutions=False, raise_exception_on_nonoptimal_result=False, tee=TEE,
                                        solver_options={"NonConvex": 2, "MIPGap": CERT_GAP, "TimeLimit": CERT_TIME})
        tc = str(res.termination_condition).split(".")[-1]; lb = float(res.objective_bound)
        ub = float(res.incumbent_objective) if res.incumbent_objective is not None else float("nan")
    secs = time.time() - t0
    gap_abs = ub - lb if np.isfinite(ub) and np.isfinite(lb) else float("nan")
    gap_rel = gap_abs / abs(ub) if np.isfinite(gap_abs) and ub != 0 else float("nan")
    ok = st == "optimal"            # timeout rows carry only a lower bound as label, so f-NN-delta < 0 there is not a violation
    chk = float((yv - eval64(onnx.load(MODEL_FILE).graph, X) - lb)[ok].min()) if ok.any() and np.isfinite(lb) else float("nan")
    out = dict(scenario=SS, tag=TAG, activation=ACTIVATION, solver=CERT_SOLVER, time_limit=CERT_TIME, rel_gap_target=CERT_GAP, termination=tc, seconds=round(secs, 1),
               delta=lb, incumbent=ub, abs_gap=gap_abs, rel_gap=gap_rel, n_training_rows=len(rows), n_optimal_label_rows=int(ok.sum()),
               check_min_f_minus_NN_minus_delta_on_optimal_rows=chk, weights_dtype="float64", max_abs_weight=wmax, input_names=in_names,
               cert_box=CERT_BOX, box_lo=lo.tolist(), box_hi=hi.tolist(), training_box_lo=tr_lo.tolist(), training_box_hi=tr_hi.tolist(), model=MODEL_FILE)
    json.dump(out, open(CERT_FILE, "w"), indent=1)
    print(f"##### scenario {SS} DONE ({secs:.0f}s): termination {tc} | delta = proven LB = {lb:.6g} | incumbent UB = {ub:.6g} | abs gap {gap_abs:.4g} | "
          f"rel gap {100 * gap_rel:.3g}% | min[f-NN-delta] on optimal-label rows = {chk:.4g} (>=0 expected)\n  result written to {CERT_FILE}", flush=True)


if RUN_SAMPLE: stage_sample()
if RUN_TRAIN: stage_train()
if RUN_CERTIFY: stage_certify()
print(f"scenario {SS}: pipeline finished in {time.time() - T_START:.0f}s", flush=True)
