"""
Infeasibility (feasibility) cuts for EDUnits_nocp from a NN that approximates the constraint violation of each scenario.

For every scenario s (= ED unit s) the script does, in this order:

  1. TRAIN     a NN_s(z) on sampled data  (z = the 4 free first-stage variables of the scenario = molN, molW of the unit's two OUTLET streams,
               label = CONTINUOUS violation v:  v <= 0 feasible,  v > 0 infeasible, larger = more infeasible).  Regression (MSE) on v, so infeasible
               samples are used instead of thrown away.  Written as a float64 ONNX file (input/output scaling folded in: raw z -> raw v).
  2. CERTIFY   solve the validation problem with a GLOBAL solver (BARON):

                   b_s = max_{y, x_s}  NN_s(z)     s.t.  all stage-1 constraints (mixers, purity, concentration bounds, ...)
                                                          all constraints of unit s (second stage),
                                                          z = 4 first-stage variables inside the scenario's own variable bounds
                                                          (NOT the training box: that is the point of the validation problem).

               b_s is the largest NN score of any truly feasible point, so
                                  NN_s(z) <= b_s          holds for EVERY feasible point (y, x_s)           (the feasibility cut).
               It only involves first-stage variables, so it is valid for the whole problem (every scenario / every B&B node below the root).
               Solved as  min -NN_s(z);  b_s = -(BARON's proven lower bound of that) = the UPPER bound of the max problem, so it is valid even if
               BARON stops at the time limit (the incumbent would NOT be safe: a larger NN value may exist).  Label quality only affects how tight b_s is.
  3. OUTPUT    <OUT_DIR>/infea_<s>.json   (cut of scenario s: input names, b, incumbent, gap, termination, bounds of z, network file)
               <OUT_DIR>/infea_<s>.onnx   (the network)
               <OUT_DIR>/infea_cuts.json  (all scenarios found so far, collected)

Usage:   python infea_contour.py            all scenarios in SCENARIOS
         python infea_contour.py 2 3        only scenarios 2 and 3
Stages can be switched off with RUN_TRAIN / RUN_CERTIFY (e.g. re-certify an existing network).

DATA INPUT IS NOT WRITTEN YET: see load_data() below (it must return the 4 inputs and the continuous violation for one scenario).
"""
import os, sys, json, time, copy, importlib, math
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
import numpy as np
import pyomo.environ as pyo

# ============================== SETTINGS (edit these) ==============================
PROBLEM_MODULE = "NSPLIB.src.instances.EDUnits_nocp.EDUnits_nocp"
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))     # networks and cuts are written here (project folder)
SCENARIOS = [1, 2, 3]
SCENARIO_OUTLETS = {1: [2, 3], 2: [5, 6], 3: [7, 8]}       # NN inputs z = molN, molW of the dilute and concentrate outlet stream of the unit
RUN_TRAIN, RUN_CERTIFY = True, True

# --- training ---
HIDDEN     = [4, 4]
ACTIVATION = "relu"                 # "relu" (OMLT big-M, MILP) or "sigmoid" (OMLT full-space smooth NLP); BARON has no tanh
LR, BATCH_SIZE, EPOCHS, PATIENCE, VAL_FRAC, SEED = 1e-3, 16, 2000, 200, 0.2, 0
TRAIN_TIME = None                   # wall-clock limit for training in seconds (None = only EPOCHS / PATIENCE)
MIN_ROWS   = 10
FEAS_TOL   = 0.0                    # a data point counts as feasible when its violation <= FEAS_TOL (used only for the diagnostics printed after certification)

# --- certification (global solver) ---
CERT_TIME = 60                      # seconds per scenario
CERT_GAP  = 1e-2                    # relative gap target (b_s is valid whatever the final gap is)
BARON_EXE = None                    # None -> found on PATH
BARON_LIC = None                    # None -> default license
TEE       = False                   # True: show the solver's live log
# ====================================================================================

# OMLT propagates pre-activation bounds numerically with exp(); wide bounds times large weights can overflow (math range error).
# Clamp the exponent for plain numbers only (bounds); symbolic expressions -- the actual constraints -- are untouched.
import omlt.neuralnet.activations.smooth as _sm, omlt.neuralnet.nn_formulation as _nf
_sig_sym = _sm.sigmoid_activation_function
def _sig_safe(x):
    if isinstance(x, (int, float, np.floating)): return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, float(x)))))
    return _sig_sym(x)
_sm.sigmoid_activation_function = _sig_safe; _nf.sigmoid_activation_function = _sig_safe

assert ACTIVATION in ("relu", "sigmoid")
REPO = os.path.dirname(os.path.abspath(__file__))
INSTANCE = PROBLEM_MODULE.split(".")[-1]                    # "EDUnits_nocp"
def in_names_of(s): return [f"mol{x}Stream{k}" for k in SCENARIO_OUTLETS[s] for x in "NW"]
def model_file(s): return os.path.join(OUT_DIR, f"infea_{s}.onnx")
def cut_file(s):   return os.path.join(OUT_DIR, f"infea_{s}.json")
def num(v): return None if v is None or not np.isfinite(v) else float(v)


# ======================================== DATA INPUT (to be written) ========================================
def load_data(s, in_names):
    """TO BE WRITTEN once the data format is decided.

    Must return for scenario s:
        X : float array (n, 4)  -- the 4 free first-stage variables, columns in the order of `in_names`
                                   (molN, molW of the dilute outlet stream, then molN, molW of the concentrate outlet stream)
        v : float array (n,)    -- CONTINUOUS violation of that point: v <= 0 feasible, v > 0 infeasible (larger = more infeasible),
                                   e.g. the optimal value t* of the phase-1 problem  min t  s.t. g(y, x) <= t  with the 4 variables fixed.
    Infeasible points are NOT to be dropped: they are the training data that carries the information on where the infeasible region is.
    """
    raise NotImplementedError("data input is not written yet: implement load_data(s, in_names) in infea_contour.py")


# ======================================== model of one scenario ========================================
def build_model(s):
    """Fresh Pyomo model of the instance with the stage-1 constraints and ONLY the constraints of unit s active, original objective off."""
    sys.path.insert(0, REPO); os.chdir(REPO)                    # NSPLIB writes log files relative to the repo root
    sm = importlib.import_module(PROBLEM_MODULE).const_model()
    m = sm.pyomo_model; scenarios = list(sm.scenarios); assert s in scenarios
    prefix = lambda t: "g_0_scenario_" if t == 0 else f"g_{t}_"
    all_cons = list(m.component_objects(pyo.Constraint, active=True))
    SC = {t: [c for c in all_cons if c.name.startswith(prefix(t))] for t in scenarios}
    SC1 = [c for c in all_cons if c.name.startswith("g_0_") and not c.name.startswith("g_0_scenario_")]     # stage-1 constraints stay active
    assert sum(len(v) for v in SC.values()) + len(SC1) == len(all_cons), "constraint grouping mismatch"
    m.obj.deactivate()
    for t in scenarios:
        for c in SC[t]: c.activate() if t == s else c.deactivate()
    return sm, m, len(SC1), len(SC[s])


def baron_solver():
    return pyo.SolverFactory("baron", executable=BARON_EXE) if BARON_EXE else pyo.SolverFactory("baron")


def eval64(graph, Xin):
    """Exported network evaluated in float64 straight from the ONNX weights (what the solver sees)."""
    from onnx import numpy_helper
    W = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in graph.initializer}
    h = Xin.astype(np.float64)
    for nd in graph.node:
        if nd.op_type == "Gemm":
            tb = next((a.i for a in nd.attribute if a.name == "transB"), 0); Wm = W[nd.input[1]]; h = h @ (Wm.T if tb else Wm) + W[nd.input[2]]
        elif nd.op_type == "Sigmoid": h = 1.0 / (1.0 + np.exp(-h))
        elif nd.op_type == "Relu": h = np.maximum(h, 0.0)
        else: raise RuntimeError(f"unexpected ONNX node {nd.op_type}")
    return h.ravel()


# ======================================== STAGE 1: training ========================================
def stage_train(s):
    import torch, torch.nn as nn, onnx
    from torch.utils.data import TensorDataset, DataLoader
    names = in_names_of(s)
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(1)            # CPU only
    X, v = load_data(s, names); X = np.asarray(X, dtype=np.float32); v = np.asarray(v, dtype=np.float32).ravel()
    assert X.ndim == 2 and X.shape[1] == len(names) and len(X) == len(v), f"load_data must return X (n,{len(names)}) and v (n,)"
    ok = np.isfinite(X).all(1) & np.isfinite(v); X, v = X[ok], v[ok]
    print(f"[train] scenario {s}: {len(v)} rows | feasible (v <= {FEAS_TOL:g}) {(v <= FEAS_TOL).sum()}, infeasible {(v > FEAS_TOL).sum()} | "
          f"violation range [{v.min():.6g}, {v.max():.6g}], std {v.std():.4g}\n  inputs {names}", flush=True)
    if len(v) < MIN_ROWS: sys.exit(f"only {len(v)} usable rows (< {MIN_ROWS}); not training")
    perm = np.random.RandomState(SEED).permutation(len(v)); nva = max(1, int(np.ceil(VAL_FRAC * len(v))))
    va_i, tr_i = perm[:nva], perm[nva:]
    Xtr, Xva, ytr, yva = X[tr_i], X[va_i], v[tr_i], v[va_i]
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
        if va < best - 1e-9: best, bad = va, 0; best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else: bad += 1
        if bad >= PATIENCE or (TRAIN_TIME and time.time() - t0 > TRAIN_TIME): break
    net.load_state_dict(best_state); net.eval()
    print(f"  trained {ep + 1} epochs in {time.time() - t0:.0f}s (best standardised val MSE {best:.4g})", flush=True)
    # export: scaling folded into first / last Linear, everything in float64 (float32 rounding of folded weights shifts outputs)
    ex = copy.deepcopy(net).double().eval(); lins = [l for l in ex if isinstance(l, nn.Linear)]
    with torch.no_grad():
        first, last = lins[0], lins[-1]; sx, mx = torch.tensor(x_std, dtype=torch.float64), torch.tensor(x_mean, dtype=torch.float64)
        first.bias.copy_(first.bias - first.weight @ (mx / sx)); first.weight.copy_(first.weight / sx)
        last.weight.mul_(y_std); last.bias.copy_(last.bias * y_std + y_mean)
    kw = dict(input_names=["input"], output_names=["output"], opset_version=13, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    dummy = torch.zeros(1, X.shape[1], dtype=torch.float64); mf = model_file(s); os.makedirs(OUT_DIR, exist_ok=True)
    try: torch.onnx.export(ex, dummy, mf, dynamo=False, **kw)       # newer PyTorch: legacy exporter -> plain Gemm nodes (what OMLT reads)
    except TypeError: torch.onnx.export(ex, dummy, mf, **kw)        # older PyTorch has no `dynamo` argument
    onnx.checker.check_model(onnx.load(mf)); g = onnx.load(mf).graph
    net64 = copy.deepcopy(net).double().eval()
    with torch.no_grad(): ref64 = net64(torch.tensor((X.astype(np.float64) - x_mean) / x_std)).numpy().ravel() * y_std + y_mean
    scale = max(1.0, float(np.abs(v).max())); err = float(np.abs(eval64(g, X) - ref64).max())
    assert err <= 1e-3 * scale, f"ONNX export mismatch (float64 evaluation): {err}"
    tr_pred, va_pred = eval64(g, Xtr), eval64(g, Xva)
    print(f"[train] scenario {s} DONE: train RMSE {np.sqrt(np.mean((tr_pred - ytr) ** 2)):.4g} | val RMSE {np.sqrt(np.mean((va_pred - yva) ** 2)):.4g} | "
          f"violation std {v.std():.4g} | ONNX check {err:.3g}\n  network written to {mf}", flush=True)


# ======================================== STAGE 2: certification ========================================
def stage_certify(s):
    import onnx
    from omlt import OmltBlock
    from omlt.io.onnx import load_onnx_neural_network
    from omlt.neuralnet import ReluBigMFormulation, FullSpaceSmoothNNFormulation
    names = in_names_of(s); mf = model_file(s)
    if not os.path.exists(mf): sys.exit(f"file not found: {mf}")
    dt = {t.data_type for t in onnx.load(mf).graph.initializer}
    if dt != {onnx.TensorProto.DOUBLE}: sys.exit(f"{mf} has weights of ONNX data type(s) {sorted(dt)}, expected float64 (11). Retrain.")
    sm, m, n1, ns = build_model(s)
    lo = np.array([m.y[n].bounds[0] for n in names], dtype=float); hi = np.array([m.y[n].bounds[1] for n in names], dtype=float)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), "network inputs need finite bounds"
    print(f"[certify] scenario {s}: {n1} stage-1 constraints + {ns} unit-{s} constraint groups active; z over the scenario's own bounds " +
          ", ".join(f"{n}[{a:.6g},{b:.6g}]" for n, a, b in zip(names, lo, hi)), flush=True)
    net = load_onnx_neural_network(onnx.load(mf), input_bounds={i: (float(lo[i]), float(hi[i])) for i in range(len(names))})
    form = ReluBigMFormulation(net) if ACTIVATION == "relu" else FullSpaceSmoothNNFormulation(net)
    m.nn = OmltBlock(); m.nn.build_formulation(form)
    m.link = pyo.ConstraintList()
    for i, n in enumerate(names): m.link.add(m.nn.inputs[i] == m.y[n])               # NN input i == first-stage variable
    m.cert_obj = pyo.Objective(expr=-m.nn.outputs[0], sense=pyo.minimize)             # min -NN  <=>  max NN; BARON's lower bound of this gives b_s = -LB
    print(f"--- scenario {s}: max NN_s(z) over the feasible set with baron (limit {CERT_TIME}s, gap {CERT_GAP:g}, {ACTIVATION} net {mf}) ---", flush=True)
    opts = {"MaxTime": CERT_TIME, "EpsR": CERT_GAP, "EpsA": 1e-6, "TDo": 0, "AbsConFeasTol": 1e-2}
    if BARON_LIC: opts["LicName"] = BARON_LIC
    t0 = time.time()
    res = baron_solver().solve(m, tee=TEE, load_solutions=False, options=opts)
    secs = time.time() - t0
    tc = str(res.solver.termination_condition).split(".")[-1]
    lb = res.problem.lower_bound; lb = float(lb) if lb is not None and abs(lb) < 1e20 else float("-inf")     # proven lower bound of min -NN
    ub = res.problem.upper_bound; ub = float(ub) if ub is not None and abs(ub) < 1e20 else float("nan")      # incumbent of min -NN
    b_cut = -lb if np.isfinite(lb) else None                          # b_s: proven UPPER bound of max NN  (None -> no usable cut)
    inc = -ub if np.isfinite(ub) else None                            # incumbent of max NN (a LOWER bound of b*: NOT safe as a cut)
    infeasible = tc in ("infeasible", "invalidProblem")
    if infeasible: b_cut = None
    gap_abs = (b_cut - inc) if (b_cut is not None and inc is not None) else None
    gap_rel = gap_abs / abs(b_cut) if gap_abs is not None and b_cut != 0 else None
    note = None
    if infeasible: note = "the scenario subproblem (stage-1 + unit constraints) is infeasible inside the variable bounds: no feasible point exists, no cut computed"
    elif b_cut is None: note = "BARON returned no finite upper bound: no usable cut (raise CERT_TIME)"
    elif gap_rel is not None and gap_rel > CERT_GAP: note = "BARON stopped with a gap: b is its proven upper bound, valid but looser than the true maximum"
    # diagnostics on the training data (only if the data loader exists): feasible points must satisfy NN <= b; infeasible points above b are cut off
    diag = {}
    try:
        X, v = load_data(s, names); X = np.asarray(X, dtype=float); v = np.asarray(v, dtype=float).ravel()
        pred = eval64(onnx.load(mf).graph, X); feas = v <= FEAS_TOL
        inside = np.all((X >= lo - 1e-9) & (X <= hi + 1e-9), axis=1)
        if b_cut is not None:
            diag = dict(n_feasible=int(feas.sum()), n_feasible_with_NN_above_b=int((feas & inside & (pred > b_cut + 1e-9)).sum()),
                        n_infeasible=int((~feas).sum()), n_infeasible_cut_off=int((~feas & (pred > b_cut)).sum()), max_NN_on_feasible_data=float(pred[feas].max()) if feas.any() else None)
    except NotImplementedError: pass
    out = dict(scenario=s, cut="NN(z) <= b", input_names=names, b=b_cut, incumbent_of_max=inc, abs_gap=gap_abs, rel_gap=gap_rel, termination=tc, seconds=round(secs, 1),
               time_limit=CERT_TIME, rel_gap_target=CERT_GAP, activation=ACTIVATION, hidden=HIDDEN, z_lo=lo.tolist(), z_hi=hi.tolist(),
               model=mf, data_check=diag, note=note)
    json.dump(out, open(cut_file(s), "w"), indent=1)
    print(f"##### scenario {s} DONE ({secs:.0f}s): termination {tc} | b = proven upper bound of max NN = " + (f"{b_cut:.6g}" if b_cut is not None else "none") +
          " | incumbent " + (f"{inc:.6g}" if inc is not None else "none") + (f" | rel gap {100 * gap_rel:.3g}%" if gap_rel is not None else ""), flush=True)
    if diag: print(f"  data check: feasible points with NN > b: {diag['n_feasible_with_NN_above_b']} (must be 0) | infeasible points cut off: {diag['n_infeasible_cut_off']} / {diag['n_infeasible']}", flush=True)
    if note: print("  NOTE: " + note, flush=True)
    print(f"  cut written to {cut_file(s)}", flush=True)


def collect_cuts():
    cuts = {}
    for s in SCENARIOS:
        if os.path.exists(cut_file(s)): cuts[str(s)] = json.load(open(cut_file(s)))
    json.dump(cuts, open(os.path.join(OUT_DIR, "infea_cuts.json"), "w"), indent=1)
    print(f"all cuts collected in {os.path.join(OUT_DIR, 'infea_cuts.json')}: scenarios {sorted(cuts)}", flush=True)


def main():
    scen = [int(a) for a in sys.argv[1:]] or SCENARIOS
    assert all(s in SCENARIO_OUTLETS for s in scen), f"scenarios must be in {sorted(SCENARIO_OUTLETS)}"
    for s in scen:
        t0 = time.time()
        if RUN_TRAIN: stage_train(s)
        if RUN_CERTIFY: stage_certify(s)
        print(f"scenario {s}: finished in {time.time() - t0:.0f}s", flush=True)
    collect_cuts()


if __name__ == "__main__":
    main()
