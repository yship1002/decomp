"""
Parallel FIXED-POINT sampling for ONE scenario (many CPUs, IPOPT).

For every point of an N_PTS^4 grid over the scenario's 4 free first-stage variables (molN, molW of its dilute and concentrate OUTLET streams,
N_PTS values spread uniformly over the ORIGINAL bounds, both end points included) the 4 variables are FIXED to the grid point and
    min f_s(y, x_s)  s.t.  stage-1 constraints, scenario-s constraints
is solved with IPOPT (the remaining first-stage variables, i.e. the inlet stream etc., and x_s are free).  Only FEASIBLE points are written
(label = local optimum of f_s at that fixed point); infeasible / failed points are only counted.

Cheap pre-screen (no solver call): a point is skipped when it violates a constraint of the model that involves ONLY the 4 fixed variables and
the inlet stream they determine (inlet = dilute + concentrate outlet): inlet bounds, concentration bound of the 3 streams, purity constraint,
and for scenario 1 the feed lower bound of mixer 1 (N1 >= Nfeed + lb(N6), W1 >= Wfeed + lb(W6)).  Skipped points are counted as "screened".

Output (OUT_DIR):  EDUnits_nocp_<s>_pts.jsonl   one line per feasible point, same fields pipeline.py reads (status, label, y of the 4 inputs, box = grid index)
                   EDUnits_nocp_<s>_pts.progress   one line per finished chunk (resume: finished chunks are skipped; delete both files to start over)
Train / certify from it with pipeline.py: set DATA_FILE accordingly (see pipeline.py DATA_SUFFIX) and RUN_SAMPLE=False.

Usage:   python sample_points.py [scenario] [n_pts]       (defaults SCENARIO / N_PTS below).  N_PROC from $SLURM_CPUS_PER_TASK if set.
"""
import os, sys, json, time, itertools, importlib, signal, multiprocessing as mp
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition as TC

# ============================== SETTINGS (edit these) ==============================
PROBLEM_MODULE = "NSPLIB.src.instances.EDUnits_nocp.EDUnits_nocp"
OUT_DIR   = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/decomp_srun"
SCENARIO  = 1                      # 1, 2 or 3 (first command-line argument overrides)
N_PTS     = 100                    # grid points per free dimension: one int or a dict keyed by variable name (second command-line argument overrides the int)
N_PROC    = 10                     # worker processes (overridden by $SLURM_CPUS_PER_TASK)
SCREEN    = True                   # skip points that violate a constraint on the fixed variables / inlet before calling the solver
MAX_POINTS = None                  # e.g. 2000 for a timing test (first MAX_POINTS grid points only); None = whole grid
SAMPLE_TIME = 60
IPOPT_OPTIONS = {"tol": 1e-6, "constr_viol_tol": 1e-2, "acceptable_tol": 1e-4, "max_iter": 3000, "max_cpu_time": SAMPLE_TIME, "print_level": 0}
CHUNK     = 2000                   # grid points per task (a chunk is the unit of resume)
LOG_EVERY = 60
SCENARIO_OUTLETS = {1: [2, 3], 2: [5, 6], 3: [7, 8]}
UNIT_INLET = {1: 1, 2: 4, 3: 5}                                              # scenario -> inlet stream (= dilute + concentrate outlet)
RANGES = {f"mol{x}Stream{k}": ((1e-5, 10.0) if x == "N" else (1e-5, 500.0)) for k in range(1, 9) for x in "NW"}
# ====================================================================================

SS = int(sys.argv[1]) if len(sys.argv) > 1 else SCENARIO
if len(sys.argv) > 2: N_PTS = int(sys.argv[2])
assert SS in SCENARIO_OUTLETS
N_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", N_PROC))
MAX_POINTS = int(os.environ["MAX_POINTS"]) if os.environ.get("MAX_POINTS") else MAX_POINTS        # env overrides for quick tests
CHUNK = int(os.environ.get("CHUNK", CHUNK))
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); sys.path.insert(0, REPO)
OUT_DIR = os.environ.get("SAMPLE_OUT", OUT_DIR)                       # env override, e.g. for a local test
OUT_DIR = OUT_DIR if os.path.isabs(OUT_DIR) else os.path.join(REPO, OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
INSTANCE = PROBLEM_MODULE.split(".")[-1]
DATA_FILE = os.path.join(OUT_DIR, f"{INSTANCE}_{SS}_pts.jsonl"); PROG_FILE = os.path.join(OUT_DIR, f"{INSTANCE}_{SS}_pts.progress")
names = [f"mol{x}Stream{k}" for k in SCENARIO_OUTLETS[SS] for x in "NW"]
npd = [int(N_PTS[n]) if isinstance(N_PTS, dict) else int(N_PTS) for n in names]; assert all(k >= 2 for k in npd)
grids = [np.linspace(RANGES[n][0], RANGES[n][1], k) for n, k in zip(names, npd)]
N_ALL = int(np.prod(npd)); N_TOTAL = min(N_ALL, MAX_POINTS) if MAX_POINTS else N_ALL
N_CHUNKS = (N_TOTAL + CHUNK - 1) // CHUNK
INLET = UNIT_INLET[SS]


def build_model():
    sm = importlib.import_module(PROBLEM_MODULE).const_model()
    m = sm.pyomo_model; scenarios = list(sm.scenarios); assert SS in scenarios
    prefix = lambda s: "g_0_scenario_" if s == 0 else f"g_{s}_"
    all_cons = list(m.component_objects(pyo.Constraint, active=True))
    SC = {s: [c for c in all_cons if c.name.startswith(prefix(s))] for s in scenarios}
    SC1 = [c for c in all_cons if c.name.startswith("g_0_") and not c.name.startswith("g_0_scenario_")]
    assert sum(len(v) for v in SC.values()) + len(SC1) == len(all_cons)
    m.obj.deactivate()
    for t in scenarios:
        for c in SC[t]: c.activate() if t == SS else c.deactivate()
    return sm, m


SM, M = build_model()
M.f_obj = pyo.Objective(expr=SM.obj_expr[SS], sense=pyo.minimize)
SOLVER = pyo.SolverFactory("ipopt")
if not SOLVER.available(exception_flag=False): sys.exit("Pyomo cannot find the ipopt executable on PATH")
# constants of the screen (read from the model; same formulas as the stage-1 constraints)
# constants copied from EDUnits_nocp.py (densityH2OGM3, molweightN, molweightH2OgMol, flowFeed, concFeed)
RHO, MN, MW = 1000 * 1000, 14.0067, 18.01528
FLOW_FEED, CONC_FEED = 192555 / (1000 * 24 * 60 * 60), 1366
KUB = 40000 * MW / (RHO - 40000 * MN)                                   # conConcUBStream*:  molN <= KUB * molW
KP1, KP3 = 400 * MW / MN / 1e6, 30000 * MW / MN / 1e6                   # conPurityED1: N2 <= KP1 W2 ; conPurityED3: N7 >= KP3 W7
NFEED, WFEED = FLOW_FEED * CONC_FEED / MN, FLOW_FEED * (RHO - CONC_FEED) / MW          # molNFeed, molWFeed
NB, WB = RANGES["molNStream1"], RANGES["molWStream1"]


def screen(P):
    """P: (n,4) points (N_a, W_a, N_b, W_b).  Boolean mask of points that pass the necessary conditions."""
    Na, Wa, Nb, Wb = P.T; Ni, Wi = Na + Nb, Wa + Wb
    ok = (Ni <= NB[1]) & (Wi <= WB[1]) & (Na <= KUB * Wa) & (Nb <= KUB * Wb) & (Ni <= KUB * Wi)
    if SS == 1: ok &= (Na <= KP1 * Wa) & (Ni >= NFEED + NB[0]) & (Wi >= WFEED + WB[0])      # mixer 1: N1 = Nfeed + N6, N6 >= 1e-5
    if SS == 3: ok &= (Na >= KP3 * Wa)                                                      # stream 7 = first outlet of scenario 3
    return ok


def solve_point(p):
    for n, v in zip(names, p): M.y[n].fix(float(v))
    M.y[f"molNStream{INLET}"].set_value(float(p[0] + p[2])); M.y[f"molWStream{INLET}"].set_value(float(p[1] + p[3]))      # start the inlet at the balance value
    status, label = "error", None
    try:
        r = SOLVER.solve(M, tee=False, load_solutions=False, options=dict(IPOPT_OPTIONS)); tc = r.solver.termination_condition
        if tc in (TC.optimal, TC.globallyOptimal, TC.locallyOptimal): M.solutions.load_from(r); status, label = "optimal", float(pyo.value(M.f_obj))
        elif tc in (TC.infeasible, TC.invalidProblem): status = "infeasible"
        else: status = "timeout_nolb"
    except Exception as e: status = "error:" + type(e).__name__
    return status, label


def do_chunk(c):
    """Grid points c*CHUNK .. (c+1)*CHUNK-1: returns (chunk, counts, JSON lines of the feasible points)."""
    a, b = c * CHUNK, min((c + 1) * CHUNK, N_TOTAL)
    ix = np.array(np.unravel_index(np.arange(a, b), npd)).T                       # (n,4) grid indices
    P = np.stack([grids[d][ix[:, d]] for d in range(4)], axis=1)
    keep = screen(P) if SCREEN else np.ones(len(P), bool)
    counts = {"screened": int((~keep).sum())}; lines = []
    for k in np.nonzero(keep)[0]:
        t0 = time.time(); st, lab = solve_point(P[k]); counts[st] = counts.get(st, 0) + 1
        if st == "optimal": lines.append(json.dumps(dict(scenario=SS, vars=names, box=ix[k].tolist(), status="optimal", label=lab, label_type="ipopt_fixed_point",
                                                          y={n: float(v) for n, v in zip(names, P[k])}, seconds=round(time.time() - t0, 3))))
    return c, counts, lines


def read_done():
    done = {}
    if os.path.exists(PROG_FILE):
        for line in open(PROG_FILE):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("n_pts") != npd or r.get("chunk_size") != CHUNK: sys.exit(f"{PROG_FILE} was written with a different grid / chunk size; move or delete it and {DATA_FILE}")
            done[r["chunk"]] = r["counts"]
    return done


def main():
    done = read_done(); todo = [c for c in range(N_CHUNKS) if c not in done]
    tot = {}
    for cnt in done.values():
        for k, v in cnt.items(): tot[k] = tot.get(k, 0) + v
    print(f"[points] scenario {SS}: grid {'x'.join(map(str, npd))} = {N_ALL} points ({N_TOTAL} used), {N_CHUNKS} chunks of {CHUNK}, {len(done)} already done, {N_PROC} processes", flush=True)
    t_all = t_log = time.time(); k = 0
    fd, fp = open(DATA_FILE, "a"), open(PROG_FILE, "a")
    with mp.get_context("fork").Pool(N_PROC) as pool:
        def _stop(*a): raise SystemExit("SIGTERM received: stopping (finished chunks are saved; run again to resume)")
        signal.signal(signal.SIGTERM, _stop)
        try:
            for c, counts, lines in pool.imap_unordered(do_chunk, todo):
                for ln in lines: fd.write(ln + "\n")
                fd.flush(); os.fsync(fd.fileno())
                fp.write(json.dumps(dict(chunk=c, n_pts=npd, chunk_size=CHUNK, counts=counts)) + "\n"); fp.flush(); os.fsync(fp.fileno())
                k += 1
                for kk, v in counts.items(): tot[kk] = tot.get(kk, 0) + v
                if time.time() - t_log > LOG_EVERY:
                    t_log = time.time(); el = t_log - t_all; rate = k / el
                    print(f"[points] {k}/{len(todo)} chunks in {el / 60:.1f} min (ETA {(len(todo) - k) / rate / 3600:.2f} h) totals {tot}", flush=True)
        finally:
            fd.close(); fp.close(); pool.terminate()
    print(f"[points] scenario {SS} DONE in {(time.time() - t_all) / 60:.1f} min: totals {tot}", flush=True)


if __name__ == "__main__":
    main()
