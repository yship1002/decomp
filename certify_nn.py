"""
Certification problem of ONE scenario: trained NN (ONNX) -> OMLT -> Pyomo -> BARON/Gurobi.  One SLURM array task = one scenario,
every scenario works independently (own model build, own network, own output file).

Usage:   python certify_nn.py <task_id>          task_id 0,1,2 -> scenario 1,2,3   (or set SLURM_ARRAY_TASK_ID)
SLURM:   see certify_nn.sbatch   (sbatch --array=0-2 certify_nn.sbatch)  -- submit it AFTER train_nn.py has written the networks.

Solves        delta_s = min  f_s(y, x) - NN_s(y)
over the first-stage variables y inside the NN's input box and the second-stage variables x, with scenario s's own constraints plus the
stage-1 constraints.  delta_s is the solver's last PROVEN lower bound (not the incumbent), so for every feasible point inside the box
        NN_s(y) + delta_s <= f_s(y, x).
Input box = min/max of the network's training inputs, recomputed here from the same rows train_nn.py used.

Inputs : <MODEL_DIR>/model_s<s>_<TAG>.onnx            from train_nn.py
         <DATA_DIR>/boxes64_scenario<s>.jsonl         from sample_boxes.py
Output : <OUT_DIR>/cert_<TAG>_<SOLVER>_scenario<s>.json   delta (proven lower bound), incumbent, gaps, termination, box, sanity check
"""
import os, sys, json, time, importlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
import torch
import numpy as np, onnx
from onnx import numpy_helper
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.io.onnx import load_onnx_neural_network
from omlt.neuralnet import ReluBigMFormulation, FullSpaceSmoothNNFormulation

# ============================== SETTINGS (edit these) ==============================
PROBLEM_MODULE = "NSPLIB.src.instances.EDUnits_nocp.EDUnits_nocp"
DATA_DIR  = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/decomp_srun"     # where sample_boxes.py wrote boxes64_scenario<s>.jsonl
MODEL_DIR = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/decomp_srun"     # where train_nn.py wrote model_s<s>_<TAG>.onnx
OUT_DIR   = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/decomp_srun"     # where the certification result is written
TAG        = "b64h4sig"             # model file tag (same as TAG in train_nn.py)
ACTIVATION = "sigmoid"              # "sigmoid" (OMLT full-space NLP) or "relu" (OMLT big-M MILP); must match how the net was trained
SOLVER     = "baron"                # "baron" or "gurobi"
TIME_LIMIT = 600                    # seconds
REL_GAP    = 1e-2                   # relative gap target (1%)
TEE        = False                  # True: show the solver's live log in the .out file
BARON_EXE = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/privatemodule/baron/baron"            # None -> found on PATH
BARON_LIC = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/privatemodule/baron/baronlice.txt"    # None -> default
SCENARIO_STREAMS = {1: [1, 2, 3], 2: [4, 5, 6], 3: [5, 7, 8]}      # NN inputs: molN, molW of each stream (stream 5 is shared)
# ====================================================================================

tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
if tid not in (0, 1, 2): sys.exit("usage: python certify_nn.py <0|1|2>   (task id 0,1,2 -> scenario 1,2,3)")
SS = tid + 1
assert ACTIVATION in ("sigmoid", "relu") and SOLVER in ("baron", "gurobi")
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)                      # NSPLIB writes log files relative to the repo root
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- 1. input box from the training rows (same filter as train_nn.py) ----------------
data_path = os.path.join(DATA_DIR, f"boxes64_scenario{SS}.jsonl")
model_path = os.path.join(MODEL_DIR, f"model_s{SS}_{TAG}.onnx")
for pth in (data_path, model_path):
    if not os.path.exists(pth): sys.exit(f"file not found: {pth}")
# the network must be the float64 export written by the current train_nn.py: with large input means / small spreads / large labels, a float32 file
# differs from the trained network by hundreds, and the certificate would then be for a different function
_dt = {t.data_type for t in onnx.load(model_path).graph.initializer}
if _dt != {onnx.TensorProto.DOUBLE}:
    sys.exit(f"{model_path} has weights of ONNX data type(s) {sorted(_dt)}, expected float64 (11). Retrain with the current train_nn.py.")
rows, seen = [], set()
for line in open(data_path):
    try: r = json.loads(line)
    except Exception: continue
    if r.get("status") in ("optimal", "timeout_lb") and r.get("label") is not None and r.get("y") and tuple(r["box"]) not in seen:
        seen.add(tuple(r["box"])); rows.append(r)
in_names = [f"mol{x}Stream{k}" for k in SCENARIO_STREAMS[SS] for x in "NW"]
X = np.array([[r["y"][n] for n in in_names] for r in rows]); yv = np.array([r["label"] for r in rows]); st = np.array([r["status"] for r in rows])
lo, hi = X.min(0), X.max(0); pad = 1e-9 * np.maximum(np.abs(hi), 1.0); lo, hi = lo - pad, hi + pad
_wmax = max(float(np.abs(numpy_helper.to_array(t)).max()) for t in onnx.load(model_path).graph.initializer)
print(f"scenario {SS}: network weights float64, largest |weight or bias| = {_wmax:.4g}", flush=True)
print(f"scenario {SS}: {len(rows)} training rows -> NN input box " + ", ".join(f"{n}[{a:.6g},{b:.6g}]" for n, a, b in zip(in_names, lo, hi)), flush=True)

# ---------------- 2. Pyomo model of the instance: only scenario SS (+ stage-1 constraints) active ----------------
sm = importlib.import_module(PROBLEM_MODULE).const_model()
m = sm.pyomo_model
scenarios = list(sm.scenarios); assert SS in scenarios
def _prefix(s): return "g_0_scenario_" if s == 0 else f"g_{s}_"
all_cons = list(m.component_objects(pyo.Constraint, active=True))
SC = {s: [c for c in all_cons if c.name.startswith(_prefix(s))] for s in scenarios}
SC1 = [c for c in all_cons if c.name.startswith("g_0_") and not c.name.startswith("g_0_scenario_")]       # stage-1 constraints stay active
assert sum(len(v) for v in SC.values()) + len(SC1) == len(all_cons), "constraint grouping mismatch"
m.obj.deactivate()
for t in scenarios:
    for c in SC[t]: c.activate() if t == SS else c.deactivate()

# ---------------- 3. NN -> OMLT block, linked to the first-stage variables ----------------
net = load_onnx_neural_network(onnx.load(model_path), input_bounds={i: (float(lo[i]), float(hi[i])) for i in range(len(in_names))})
form = ReluBigMFormulation(net) if ACTIVATION == "relu" else FullSpaceSmoothNNFormulation(net)
m.nn = OmltBlock(); m.nn.build_formulation(form)
m.link = pyo.ConstraintList()
for i, n in enumerate(in_names): m.link.add(m.nn.inputs[i] == m.y[n])          # NN input i == first-stage variable (stream 5 is shared by scenarios 2 and 3)
m.cert_obj = pyo.Objective(expr=sm.obj_expr[SS] - m.nn.outputs[0], sense=pyo.minimize)          # f_s - NN_s

# ---------------- 4. solve ----------------
print(f"--- certification scenario {SS}: min f_s - NN_s with {SOLVER} (limit {TIME_LIMIT}s, gap {REL_GAP:g}, {ACTIVATION} net {model_path}) ---", flush=True)
t0 = time.time()
if SOLVER == "baron":
    opts = {"MaxTime": TIME_LIMIT, "EpsR": REL_GAP, "EpsA": 1e-6}
    if BARON_LIC: opts["LicName"] = BARON_LIC
    opt = pyo.SolverFactory("baron", executable=BARON_EXE) if BARON_EXE else pyo.SolverFactory("baron")
    res = opt.solve(m, tee=TEE, load_solutions=False, options=opts)
    tc = str(res.solver.termination_condition).split(".")[-1]; lb = float(res.problem.lower_bound)
    ub = res.problem.upper_bound; ub = float(ub) if ub is not None and abs(ub) < 1e20 else float("nan")
else:
    from pyomo.contrib.solver.solvers.gurobi_direct_minlp import GurobiDirectMINLP     # new-API Gurobi with nonlinear support
    res = GurobiDirectMINLP().solve(m, load_solutions=False, raise_exception_on_nonoptimal_result=False, tee=TEE,
                                    solver_options={"NonConvex": 2, "MIPGap": REL_GAP, "TimeLimit": TIME_LIMIT})
    tc = str(res.termination_condition).split(".")[-1]; lb = float(res.objective_bound)
    ub = float(res.incumbent_objective) if res.incumbent_objective is not None else float("nan")
secs = time.time() - t0
gap_abs = ub - lb if np.isfinite(ub) and np.isfinite(lb) else float("nan")
gap_rel = gap_abs / abs(ub) if np.isfinite(gap_abs) and ub != 0 else float("nan")

# ---------------- 5. sanity check: f - NN - delta >= 0 on the rows with exact (optimal) labels ----------------
_g = onnx.load(model_path).graph; _W = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in _g.initializer}
def nn_eval(Xin):                                              # the exported network in float64 (what the solver sees)
    h = Xin.astype(np.float64)
    for nd in _g.node:
        if nd.op_type == "Gemm":
            tb = next((a.i for a in nd.attribute if a.name == "transB"), 0); Wm = _W[nd.input[1]]; h = h @ (Wm.T if tb else Wm) + _W[nd.input[2]]
        elif nd.op_type == "Sigmoid": h = 1.0 / (1.0 + np.exp(-h))
        elif nd.op_type == "Relu": h = np.maximum(h, 0.0)
        else: raise RuntimeError(f"unexpected ONNX node {nd.op_type}")
    return h.ravel()
ok = st == "optimal"            # timeout rows carry only a lower bound as label, so f-NN-delta < 0 there is not a violation
chk = float((yv - nn_eval(X) - lb)[ok].min()) if ok.any() and np.isfinite(lb) else float("nan")

out = dict(scenario=SS, tag=TAG, activation=ACTIVATION, solver=SOLVER, time_limit=TIME_LIMIT, rel_gap_target=REL_GAP, termination=tc, seconds=round(secs, 1),
           delta=lb, incumbent=ub, abs_gap=gap_abs, rel_gap=gap_rel, n_training_rows=len(rows), n_optimal_label_rows=int(ok.sum()),
           check_min_f_minus_NN_minus_delta_on_optimal_rows=chk, weights_dtype="float64", max_abs_weight=_wmax, input_names=in_names, box_lo=lo.tolist(), box_hi=hi.tolist(), model=model_path)
out_path = os.path.join(OUT_DIR, f"cert_{TAG}_{SOLVER}_scenario{SS}.json")
json.dump(out, open(out_path, "w"), indent=1)
print(f"##### scenario {SS} DONE ({secs:.0f}s): termination {tc} | delta = proven LB = {lb:.6g} | incumbent UB = {ub:.6g} | abs gap {gap_abs:.4g} | "
      f"rel gap {100 * gap_rel:.3g}% | min[f-NN-delta] on optimal-label rows = {chk:.4g} (>=0 expected)\n  result written to {out_path}", flush=True)
