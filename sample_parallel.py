"""
Parallel stage-1 sampling for ONE scenario (many CPUs, IPOPT).  Same data format as pipeline.py stage 1, so pipeline.py (RUN_SAMPLE=False)
can train / certify from the file written here.

For every box of a BOXES_PER_DIM^6 grid over the scenario's 6 first-stage variables (molN, molW of its 3 streams, original bounds):
    min f_s(y, x_s)  s.t.  stage-1 constraints, scenario-s constraints, the 6 variables inside the box
and one JSON line per box is appended to <OUT_DIR>/<INSTANCE>_<s>.jsonl   (status optimal / infeasible / timeout_nolb / error:*).

Parallelism: the model is built once, then N_PROC worker processes are forked (each has its own copy); the main process is the only
writer of the output file. The run is RESUMABLE: boxes already in the file are skipped (error:* boxes are retried), so after a
walltime kill / SIGTERM just start it again.

Usage:   python sample_parallel.py [scenario]            (scenario defaults to SCENARIO below; 1, 2 or 3)
         N_PROC from $SLURM_CPUS_PER_TASK if set, else N_PROC below.
"""
import os, sys, json, time, math, itertools, importlib, signal, multiprocessing as mp
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"   # one thread per IPOPT process
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition as TC

# ============================== SETTINGS (edit these) ==============================
PROBLEM_MODULE = "NSPLIB.src.instances.EDUnits_nocp.EDUnits_nocp"
OUT_DIR   = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/decomp_srun"      # same folder as pipeline.py -> the file is EDUnits_nocp_<s>.jsonl there
SCENARIO  = 1                                                                # 1, 2 or 3 (command-line argument overrides)
N_PROC    = 10                                                               # worker processes (overridden by $SLURM_CPUS_PER_TASK)
BOXES_PER_DIM = 10                                                           # pieces per boxed dimension: one int, or a dict keyed by variable name (10 -> 10^6 boxes)
MAX_BOXES = None                                                             # e.g. 200 for a quick test; None = all boxes
SAVE_X    = False                                                            # True: also store the scenario variables x_s of every solution (file ~20x bigger); training/certification only need y
SAMPLE_TIME = 60                                                             # IPOPT cpu-time limit (s) per box
IPOPT_OPTIONS = {"tol": 1e-6, "constr_viol_tol": 1e-2, "acceptable_tol": 1e-4, "max_iter": 3000, "max_cpu_time": SAMPLE_TIME, "print_level": 0}   # same as pipeline.py
CHUNK       = 20                      # boxes handed to a worker at a time
FLUSH_EVERY = 500                     # fsync the output file every this many finished boxes
LOG_EVERY   = 60                      # progress line every this many seconds
SCENARIO_STREAMS = {1: [1, 2, 3], 2: [4, 5, 6], 3: [5, 7, 8]}               # boxed variables: molN, molW of each stream (stream 5 is shared)
RANGES = {f"mol{x}Stream{k}": ((1e-5, 10.0) if x == "N" else (1e-5, 500.0)) for k in range(1, 9) for x in "NW"}    # original variable bounds
# ====================================================================================

SS = int(sys.argv[1]) if len(sys.argv) > 1 else SCENARIO
assert SS in SCENARIO_STREAMS, f"scenario must be one of {sorted(SCENARIO_STREAMS)}"
N_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", N_PROC))
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO); sys.path.insert(0, REPO)                                      # NSPLIB writes log files relative to the repo root
OUT_DIR = OUT_DIR if os.path.isabs(OUT_DIR) else os.path.join(REPO, OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
INSTANCE  = PROBLEM_MODULE.split(".")[-1]
DATA_FILE = os.path.join(OUT_DIR, f"{INSTANCE}_{SS}.jsonl")
names = [f"mol{x}Stream{k}" for k in SCENARIO_STREAMS[SS] for x in "NW"]       # the 6 boxed variables (same order as in_names of pipeline.py)
nb = [int(BOXES_PER_DIM[n]) if isinstance(BOXES_PER_DIM, dict) else int(BOXES_PER_DIM) for n in names]; assert all(k >= 1 for k in nb)
edges = [np.linspace(RANGES[n][0], RANGES[n][1], k + 1) for n, k in zip(names, nb)]
N_BOXES = int(np.prod(nb)); N_TOTAL = min(N_BOXES, MAX_BOXES) if MAX_BOXES else N_BOXES


def num(v): return None if v is None or not np.isfinite(v) else float(v)


def build_model():
    """Model with only scenario SS (+ stage-1 constraints) active and the original objective off (as in pipeline.py)."""
    sm = importlib.import_module(PROBLEM_MODULE).const_model()
    m = sm.pyomo_model; scenarios = list(sm.scenarios); assert SS in scenarios
    prefix = lambda s: "g_0_scenario_" if s == 0 else f"g_{s}_"
    all_cons = list(m.component_objects(pyo.Constraint, active=True))
    SC = {s: [c for c in all_cons if c.name.startswith(prefix(s))] for s in scenarios}
    SC1 = [c for c in all_cons if c.name.startswith("g_0_") and not c.name.startswith("g_0_scenario_")]
    assert sum(len(v) for v in SC.values()) + len(SC1) == len(all_cons), "constraint grouping mismatch"
    m.obj.deactivate()
    for t in scenarios:
        for c in SC[t]: c.activate() if t == SS else c.deactivate()
    return sm, m


# ---------------- model + solver: built once in the parent, inherited by the forked workers ----------------
SM, M = build_model()
M.f_obj = pyo.Objective(expr=SM.obj_expr[SS], sense=pyo.minimize)
ALL_Y = list(SM.y_set)
SOLVER = pyo.SolverFactory("ipopt")
if not SOLVER.available(exception_flag=False): sys.exit("Pyomo cannot find the ipopt executable on PATH")


def solve_box(idx):
    """Solve one box in a worker process; returns (status, JSON line of the record)."""
    L = [edges[d][idx[d]] for d in range(6)]; U = [edges[d][idx[d] + 1] for d in range(6)]
    for d, n in enumerate(names):
        v = M.y[n]; v.setlb(L[d]); v.setub(U[d]); v.set_value(0.5 * (L[d] + U[d]))
    status, label, ltype, ub, yv, xv = "error", None, None, None, None, None
    t0 = time.time()
    try:
        r = SOLVER.solve(M, tee=False, load_solutions=False, options=dict(IPOPT_OPTIONS))
        tc = r.solver.termination_condition
        if tc in (TC.optimal, TC.globallyOptimal, TC.locallyOptimal):
            M.solutions.load_from(r); status, ltype = "optimal", "ipopt_local"; label = float(pyo.value(M.f_obj)); ub = label
            yv = {n: float(M.y[n].value) for n in ALL_Y}
            if SAVE_X: xv = {v.name: float(v.value) for v in M.component_data_objects(pyo.Var) if v.value is not None and not v.name.startswith("y[")}
        elif tc in (TC.infeasible, TC.invalidProblem):
            status = "infeasible"
        else:
            status = "timeout_nolb"                                   # iteration / cpu-time limit, numerical failure, unbounded: no usable point
    except Exception as e:
        status = "error:" + type(e).__name__
    rec = dict(scenario=SS, vars=names, box=list(idx), lo=[float(a) for a in L], hi=[float(a) for a in U], status=status, label=num(label),
               label_type=ltype, upper_bound=num(ub), seconds=round(time.time() - t0, 2), y=yv, x=xv)
    return status, json.dumps(rec)


def read_done():
    """Boxes already in the file (error:* rows are retried); checks that the file uses the same grid."""
    done = set()
    if not os.path.exists(DATA_FILE): return done
    for line in open(DATA_FILE):
        try: rec = json.loads(line)
        except Exception: continue                                    # partially written last line
        ix = tuple(rec["box"])
        ok = rec.get("vars", names) == names and len(ix) == 6 and all(
            0 <= ix[d] < nb[d] and math.isclose(rec["lo"][d], edges[d][ix[d]], rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(rec["hi"][d], edges[d][ix[d] + 1], rel_tol=1e-9, abs_tol=1e-12) for d in range(6))
        if not ok: sys.exit(f"{DATA_FILE} was written with a different box grid / variable set than BOXES_PER_DIM={BOXES_PER_DIM}; move or delete it and run again")
        if not str(rec.get("status", "")).startswith("error"): done.add(ix)
    return done


def main():
    done = read_done()
    all_boxes = itertools.product(*[range(k) for k in nb])
    if MAX_BOXES: all_boxes = itertools.islice(all_boxes, MAX_BOXES)
    todo = (idx for idx in all_boxes if idx not in done)
    n_left = max(N_TOTAL - len(done), 0)
    print(f"[sample] scenario {SS}: {N_BOXES} boxes ({'x'.join(map(str, nb))}), {len(done)} already in {DATA_FILE}, {N_PROC} processes, IPOPT {IPOPT_OPTIONS}", flush=True)
    counts, k, t_all, t_log = {}, 0, time.time(), time.time()
    fh = open(DATA_FILE, "a")
    ctx = mp.get_context("fork")
    with ctx.Pool(N_PROC) as pool:
        def _stop(*a): raise SystemExit("SIGTERM received: stopping (progress is saved; run again to resume)")
        signal.signal(signal.SIGTERM, _stop)                          # set after forking so the workers keep the default handler
        try:
            for st, line in pool.imap_unordered(solve_box, todo, chunksize=CHUNK):
                fh.write(line + "\n"); k += 1
                counts[st] = counts.get(st, 0) + 1
                if k % FLUSH_EVERY == 0: fh.flush(); os.fsync(fh.fileno())
                if time.time() - t_log > LOG_EVERY:
                    t_log = time.time(); el = t_log - t_all; rate = k / el
                    print(f"[sample] {k}/{n_left} boxes in {el / 60:.1f} min ({rate:.1f}/s, ETA {(n_left - k) / rate / 3600:.1f} h) {counts}", flush=True)
        finally:
            fh.flush(); os.fsync(fh.fileno()); fh.close()
            pool.terminate()
    print(f"[sample] scenario {SS} DONE: {k} boxes in {(time.time() - t_all) / 60:.1f} min: {counts}", flush=True)


if __name__ == "__main__":
    main()
