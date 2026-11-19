"""
Combine the three certified scenario networks into a lower bound for the OVERALL problem of EDUnits_nocp.

    overall problem :  min_{y, x}  sum_s f_s(y, x_s)                 s.t. stage-1 constraints and every scenario's constraints
    bound problem   :  min_{y}     sum_s ( NN_s(y) + delta_s )       s.t. stage-1 constraints, y inside every network's input box

Every scenario was certified separately:  NN_s(y) + delta_s <= f_s(y, x_s)  for all feasible (y, x_s) with y inside the box of NN_s.
Summing over the scenarios gives  sum_s f_s >= sum_s ( NN_s + delta_s ), so the optimal value of the bound problem (BARON's proven
lower bound of it) is a lower bound for the overall problem -- on the part of the stage-1 feasible set that lies inside all three boxes.
The second-stage variables disappear: only the 16 first-stage variables, the stage-1 constraints and the three networks remain.

Usage:   python combine.py            (no argument; run it AFTER pipeline.py has finished for all three scenarios)
Inputs : <DIR>/<MODEL_TEMPLATE>   trained networks (float64 ONNX)
         <DIR>/<CERT_TEMPLATE>    certification results (delta_s, input box)
Output : <DIR>/<OUT_NAME>         overall lower bound, per-scenario deltas, gaps, termination
"""
import os, sys, json, time, importlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
import numpy as np, onnx
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.io.onnx import load_onnx_neural_network
from omlt.neuralnet import ReluBigMFormulation, FullSpaceSmoothNNFormulation

# ============================== SETTINGS (edit these) ==============================
PROBLEM_MODULE = "NSPLIB.src.instances.EDUnits_nocp.EDUnits_nocp"
DIR            = os.path.dirname(os.path.abspath(__file__))        # folder with the networks and certification files (= this project folder)
MODEL_TEMPLATE = "EDUnits_nocp_{s}.onnx"                          # {s} = scenario number
CERT_TEMPLATE  = "EDUnits_nocp_{s}.json"
OUT_NAME       = "overall_lower_bound.json"
SCENARIOS      = [1, 2, 3]
ACTIVATION     = "sigmoid"          # "sigmoid" or "relu"; must match how the networks were trained
INPUT_BOX      = "original"         # "original": y ranges over the instance's ORIGINAL variable bounds (training data is ignored)
                                    # "certified": y restricted to each network's training box (min/max of its training inputs; can be infeasible)
TIME_LIMIT     = 600                # seconds
REL_GAP        = 1e-2
TEE            = False
BARON_EXE = None                    # None -> found on PATH
BARON_LIC = None                    # None -> default license
SCENARIO_OUTLETS = {1: [2, 3], 2: [5, 6], 3: [7, 8]}               # NN inputs: molN, molW of the dilute and concentrate outlet stream of each unit (disjoint between scenarios)
ADD_UNIT_BALANCES = False           # True: also add inlet = dilute outlet + concentrate outlet (N and W) for units 1,2,3 (inlet streams 1,4,5); implied by the unit models, so valid, but not in the stage-1 constraints
UNIT_INLET = {1: (2, 3), 4: (5, 6), 5: (7, 8)}
# ====================================================================================

# OMLT propagates pre-activation bounds numerically with exp(); the wide original bounds times large weights overflow it (math range error).
# Clamp the exponent for plain numbers only (bounds); symbolic expressions -- the actual constraints -- are untouched.
import math
import omlt.neuralnet.activations.smooth as _sm, omlt.neuralnet.nn_formulation as _nf
_sig_sym = _sm.sigmoid_activation_function
def _sig_safe(x):
    if isinstance(x, (int, float, np.floating)): return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, float(x)))))
    return _sig_sym(x)
_sm.sigmoid_activation_function = _sig_safe; _nf.sigmoid_activation_function = _sig_safe

assert ACTIVATION in ("sigmoid", "relu") and INPUT_BOX in ("original", "certified")
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)                           # NSPLIB writes log files relative to the repo root

# ---------------- 1. read networks and certificates ----------------
S = {}
for s in SCENARIOS:
    mp, cp = os.path.join(DIR, MODEL_TEMPLATE.format(s=s)), os.path.join(DIR, CERT_TEMPLATE.format(s=s))
    for p in (mp, cp):
        if not os.path.exists(p): sys.exit(f"file not found: {p}")
    dt = {t.data_type for t in onnx.load(mp).graph.initializer}
    if dt != {onnx.TensorProto.DOUBLE}: sys.exit(f"{mp}: weights are ONNX type(s) {sorted(dt)}, expected float64 (11)")
    c = json.load(open(cp))
    if not np.isfinite(c["delta"]): sys.exit(f"scenario {s}: certification has no finite delta (termination {c['termination']})")
    exp = [f"mol{x}Stream{k}" for k in SCENARIO_OUTLETS[s] for x in "NW"]
    if c["input_names"] != exp: sys.exit(f"scenario {s}: input names in {cp} differ from the expected {exp}")
    S[s] = dict(model=mp, delta=float(c["delta"]), lo=np.array(c["box_lo"]), hi=np.array(c["box_hi"]), cert=c, names=exp)
    print(f"scenario {s}: delta = {S[s]['delta']:.6g}  (certified gap {100 * c['rel_gap']:.3g}%, {c['termination']}), box " +
          ", ".join(f"{n}[{a:.4g},{b:.4g}]" for n, a, b in zip(exp, S[s]["lo"], S[s]["hi"])), flush=True)

# ---------------- 2. model: stage-1 constraints only ----------------
sm = importlib.import_module(PROBLEM_MODULE).const_model()
m = sm.pyomo_model
scenarios = list(sm.scenarios)
def _prefix(t): return "g_0_scenario_" if t == 0 else f"g_{t}_"
all_cons = list(m.component_objects(pyo.Constraint, active=True))
SC = {t: [c for c in all_cons if c.name.startswith(_prefix(t))] for t in scenarios}
SC1 = [c for c in all_cons if c.name.startswith("g_0_") and not c.name.startswith("g_0_scenario_")]
assert sum(len(v) for v in SC.values()) + len(SC1) == len(all_cons), "constraint grouping mismatch"
m.obj.deactivate()
for t in scenarios:
    for c in SC[t]: c.deactivate()                                  # second-stage constraints are not needed: the networks replace them
print(f"stage-1 constraints kept: {len(SC1)}", flush=True)

# ---------------- 3. networks -> OMLT blocks linked to the first-stage variables ----------------
m.link = pyo.ConstraintList(); outs = []
for s in SCENARIOS:
    if INPUT_BOX == "original": bnd = {i: tuple(float(v) for v in m.y[n].bounds) for i, n in enumerate(S[s]["names"])}
    else: bnd = {i: (float(S[s]["lo"][i]), float(S[s]["hi"][i])) for i in range(len(S[s]["names"]))}
    S[s]["used_lo"], S[s]["used_hi"] = [bnd[i][0] for i in range(len(S[s]["names"]))], [bnd[i][1] for i in range(len(S[s]["names"]))]
    assert all(np.isfinite(v) for i in range(len(S[s]["names"])) for v in bnd[i]), "network inputs need finite bounds"
    net = load_onnx_neural_network(onnx.load(S[s]["model"]), input_bounds=bnd)
    blk = OmltBlock(); setattr(m, f"nn{s}", blk)
    blk.build_formulation(ReluBigMFormulation(net) if ACTIVATION == "relu" else FullSpaceSmoothNNFormulation(net))
    for i, n in enumerate(S[s]["names"]): m.link.add(blk.inputs[i] == m.y[n])           
    outs.append(blk.outputs[0])
if ADD_UNIT_BALANCES:
    for k, (a, b) in UNIT_INLET.items():
        for x in "NW": m.link.add(m.y[f"mol{x}Stream{k}"] == m.y[f"mol{x}Stream{a}"] + m.y[f"mol{x}Stream{b}"])
DELTA_SUM = sum(S[s]["delta"] for s in SCENARIOS)
m.bound_obj = pyo.Objective(expr=sum(outs), sense=pyo.minimize)       # sum_s NN_s(y); the constant sum_s delta_s is added after the solve

# ---------------- 4. solve ----------------
print(f"--- overall bound: min sum_s NN_s(y) with baron (limit {TIME_LIMIT}s, gap {REL_GAP:g}); sum of deltas = {DELTA_SUM:.6g} ---", flush=True)
opts = {"MaxTime": TIME_LIMIT, "EpsR": REL_GAP, "EpsA": 1e-6, "TDo": 0, "AbsConFeasTol": 1e-2}
if BARON_LIC: opts["LicName"] = BARON_LIC
opt = pyo.SolverFactory("baron", executable=BARON_EXE) if BARON_EXE else pyo.SolverFactory("baron")
t0 = time.time()
res = opt.solve(m, tee=TEE, load_solutions=False, options=opts)
secs = time.time() - t0
tc = str(res.solver.termination_condition).split(".")[-1]
lb = float(res.problem.lower_bound); ub = res.problem.upper_bound; ub = float(ub) if ub is not None and abs(ub) < 1e20 else float("nan")
feasible = tc not in ("infeasible", "invalidProblem") and np.isfinite(lb) and abs(lb) < 1e20
overall_lb = lb + DELTA_SUM if feasible else float("nan")
overall_ub = ub + DELTA_SUM if np.isfinite(ub) else float("nan")

out = dict(termination=tc, seconds=round(secs, 1), time_limit=TIME_LIMIT, activation=ACTIVATION, input_box=INPUT_BOX,
           overall_lower_bound=overall_lb, nn_part_lower_bound=lb if feasible else None, sum_deltas=DELTA_SUM,
           bound_problem_incumbent_plus_deltas=overall_ub,
           per_scenario={str(s): dict(delta=S[s]["delta"], cert_termination=S[s]["cert"]["termination"], cert_rel_gap=S[s]["cert"]["rel_gap"],
                                      training_box_lo=S[s]["lo"].tolist(), training_box_hi=S[s]["hi"].tolist(), used_input_lo=S[s]["used_lo"], used_input_hi=S[s]["used_hi"], input_names=S[s]["names"]) for s in SCENARIOS})
if not feasible:
    out["note"] = ("The bound problem is infeasible: no first-stage point satisfies the stage-1 constraints AND lies inside all network input boxes. "
                   "The networks' input boxes need not agree with the stage-1 mixer / balance constraints.")
json.dump(out, open(os.path.join(DIR, OUT_NAME), "w"), indent=1)
print(f"##### overall: termination {tc} ({secs:.0f}s)")
if feasible:
    print(f"  min sum NN_s = {lb:.6g} | + sum deltas {DELTA_SUM:.6g} -> OVERALL LOWER BOUND = {overall_lb:.6g}   (bound-problem incumbent + deltas = {overall_ub:.6g})")
    print(("  y ranges over the ORIGINAL variable bounds: each delta_s was certified only inside that network's training box, so outside it the bound is not certified; "
           if INPUT_BOX == "original" else "  valid for stage-1-feasible y inside all three network boxes; ") + "bound-problem solver gap "
          + ("%.3g%%" % (100 * (ub - lb) / abs(ub)) if np.isfinite(ub) and ub != 0 else "n/a"))
else:
    print("  " + out["note"])
print(f"  written to {os.path.join(DIR, OUT_NAME)}", flush=True)
