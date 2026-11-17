"""
Box sampling of one scenario of EDUnits_nocp (one SLURM array task = one scenario).

Usage:   python sample_boxes.py <task_id>          task_id 0,1,2  ->  scenario 1,2,3   (or set SLURM_ARRAY_TASK_ID)
SLURM:   see sample_boxes.sbatch   (sbatch --array=0-2 sample_boxes.sbatch)

What it does
  * The 6 free first-stage coordinates (N6, W6, N8, W8, N2, W2) each span an interval [lo, hi] (SETTINGS below).
    Every interval is split into two boxes (lo, mid) and (mid, hi)  ->  2^6 = 64 boxes.
  * For every box it solves ONE subproblem: scenario s's own problem (its second-stage constraints + the stage-1 constraints,
    objective = the scenario's cost f_s) with the 6 coordinates restricted to the box and all other first-stage and
    second-stage variables free (BARON, short time limit).
  * Result of each subproblem is APPENDED to  <OUT_DIR>/boxes64_scenario<s>.jsonl  as soon as it is done (one JSON line per box),
    so a killed/timed-out job keeps everything finished so far, and a re-submitted job skips boxes already in the file.

One line of the output file:
  box (list of 6 indices 0/1), lo/hi (6 bounds), status, label, label_type, upper_bound, seconds,
  y (the 16 first-stage variables at BARON's solution), x (all other variables at the solution)
  status:  "optimal"      BARON met the gap  -> label = optimal objective
           "timeout_lb"   time limit, BARON had a bound -> label = last proven LOWER bound (y, x = incumbent if BARON had one)
           "infeasible"   proven infeasible -> no label (discard for training)
           "timeout_nolb" / "error:..." -> no label
"""
import os, sys, time, json, itertools, importlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition as TC

# ============================== SETTINGS (edit these) ==============================
PROBLEM_MODULE = "NSPLIB.src.instances.EDUnits_nocp.EDUnits_nocp"
BARON_EXE = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/privatemodule/baron"            # cluster: path of the baron executable (None -> found on PATH)
BARON_LIC = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/privatemodule/baronlice.txt"    # cluster: path of the license file (None -> default)
BARON_OPTIONS = {"MaxTime": 30, "EpsR": 1e-2, "EpsA": 1e-6}   # time limit (s) per subproblem, 1% relative gap
OUT_DIR = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/decomp_srun"                             # relative to this file's folder (or absolute)
# The 6 free dimensions (N6, W6, N8, W8, N2, W2) and the interval (lo, hi) each one spans. NONE of them is fixed:
# every interval is split into (lo, mid) and (mid, hi) below, so all six are divided into two boxes (2^6 = 64 boxes).
#   N6, W6, N8, W8: the ORIGINAL variable bounds of the instance (no stored solution is used anywhere).
#   N2, W2: the exact range the stage-1 constraints allow (min/max of each variable over the stage-1 constraints + variable bounds +
#           unit balances, solved with BARON): N2 in [1e-5, 0.0635099], W2 in [117.907, 123.446] (original bounds were [1e-5, 10] and [1e-5, 500]).
#           The binding constraints are conPurityED1 (N2 <= 5.145e-4*W2) and conPurityED3 (N7 >= 0.03859*W7, N7 = Nf-N2, W7 = Wf-W2);
#           outside these ranges every point is infeasible. The feasible set in (N2, W2) is a slanted wedge, so some box corners are still infeasible.
#           To use the original bounds for N2/W2 too, set them to (1e-5, 10) and (1e-5, 500).
RANGES = {"N6": (1e-5, 10.0), "W6": (1e-5, 500.0), "N8": (1e-5, 10.0), "W8": (1e-5, 500.0),
          "N2": (1e-5, 0.0635099), "W2": (117.907, 123.446)}
COORD_VARS = {"N6": "molNStream6", "W6": "molWStream6", "N8": "molNStream8", "W8": "molWStream8", "N2": "molNStream2", "W2": "molWStream2"}
MAX_BOXES = None                                             # e.g. 4 for a quick test; None = all 64
# ====================================================================================

# ---- task id -> scenario ----
tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
if tid not in (0, 1, 2): sys.exit("usage: python sample_boxes.py <0|1|2>   (task id 0,1,2 -> scenario 1,2,3)")
SS = tid + 1                                                 # scenario number used by the instance (1,2,3)

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)                     # NSPLIB writes log files relative to the repo root
OUT_DIR = OUT_DIR if os.path.isabs(OUT_DIR) else os.path.join(REPO, OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, f"boxes64_scenario{SS}.jsonl")

# ---- build the Pyomo model, keep only scenario SS (+ stage-1 constraints) active ----
sm = importlib.import_module(PROBLEM_MODULE).const_model()
m = sm.pyomo_model
all_y = list(sm.y_set)
scenarios = list(sm.scenarios)
assert SS in scenarios, f"scenario {SS} not in {scenarios}"
def _prefix(s): return "g_0_scenario_" if s == 0 else f"g_{s}_"
all_cons = list(m.component_objects(pyo.Constraint, active=True))
SC = {s: [c for c in all_cons if c.name.startswith(_prefix(s))] for s in scenarios}
SC1 = [c for c in all_cons if c.name.startswith("g_0_") and not c.name.startswith("g_0_scenario_")]   # stage-1 constraints (stay active)
assert sum(len(v) for v in SC.values()) + len(SC1) == len(all_cons), "constraint grouping mismatch"
m.obj.deactivate()
m.f_obj = pyo.Objective(expr=sm.obj_expr[SS], sense=pyo.minimize)
for t in scenarios:
    for c in SC[t]: c.activate() if t == SS else c.deactivate()

opts = dict(BARON_OPTIONS);
if BARON_LIC: opts["LicName"] = BARON_LIC
baron = pyo.SolverFactory("baron", executable=BARON_EXE) if BARON_EXE else pyo.SolverFactory("baron")

# ---- the 64 boxes ----
names = list(RANGES)                                         # N6, W6, N8, W8, N2, W2
lo = np.array([RANGES[n][0] for n in names]); hi = np.array([RANGES[n][1] for n in names]); mid = 0.5 * (lo + hi)
boxes = list(itertools.product(range(2), repeat=len(names)))   # 0 = (lo, mid), 1 = (mid, hi) per dimension
if MAX_BOXES: boxes = boxes[:MAX_BOXES]

done = set()
if os.path.exists(OUT_FILE):                                 # resume: skip boxes already written
    for line in open(OUT_FILE):
        try: done.add(tuple(json.loads(line)["box"]))
        except Exception: pass
print(f"scenario {SS} (task {tid}): {len(boxes)} boxes, {len(done)} already in {OUT_FILE}", flush=True)

def num(v): return None if v is None or not np.isfinite(v) else float(v)

t_all = time.time(); counts = {}
for k, idx in enumerate(boxes):
    if idx in done: continue
    L = [lo[d] if idx[d] == 0 else mid[d] for d in range(6)]; U = [mid[d] if idx[d] == 0 else hi[d] for d in range(6)]
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
                status, ltype, label = "timeout_lb", "lower_bound", float(lb)
                ubv = r.problem.upper_bound; ub = num(ubv) if ubv is not None and abs(ubv) < 1e20 else None
                try: m.solutions.load_from(r); yv, xv = snap()
                except Exception: pass
            else: status = "timeout_nolb"
        if ub is None and status == "optimal": ub = label
    except Exception as e:
        status = "error:" + type(e).__name__
    rec = dict(scenario=SS, box=list(idx), lo=[float(a) for a in L], hi=[float(a) for a in U], status=status, label=num(label),
               label_type=ltype, upper_bound=num(ub), seconds=round(time.time() - t0, 2), y=yv, x=xv)
    with open(OUT_FILE, "a") as fh:                          # write each finished subproblem immediately
        fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
    counts[status] = counts.get(status, 0) + 1
    print(f"[{k + 1}/{len(boxes)}] box {idx}: {status}" + (f" label {label:.6g}" if label is not None else "") + f" ({rec['seconds']}s)", flush=True)

print(f"scenario {SS} DONE in {time.time() - t_all:.0f}s: {counts} -> {OUT_FILE}", flush=True)
