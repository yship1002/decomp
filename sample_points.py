"""
Parallel point sampling for ONE scenario (many CPUs, IPOPT).  Two modes (MODE below, default "fixed"):
  "fixed"   : fix the 4 outlet variables at each grid point, solve min f_s.  EVERY point is written: optimal -> label = cost, infeasible -> label = inf.
  "penalty" : smooth extension into the infeasible region (4 variables free, min f_s + PEN_RHO*(scaled distance to the grid point)); every point gets a finite label.

SAMPLING SPACE (SPACE = "fcr", default): an N_PTS^4 grid over (flow, conc, cf, rec) of the unit INLET (bounds QBOUNDS, log/linear spacing QSPACING, end points included).
Each grid point is converted by to_outlets() into the 4 first-stage variables of the scenario (molN, molW of its dilute and concentrate OUTLET streams), which are FIXED, and
    min f_s(y, x_s)  s.t.  stage-1 constraints, scenario-s constraints
is solved with IPOPT (the remaining first-stage variables, i.e. the inlet stream etc., and x_s are free).  There is NO pre-screen: every point goes to the solver
(only an outlet outside its original variable bounds is labelled infeasible directly, because a fixed variable is not bounds-checked).

Output (OUT_DIR):  EDUnits_nocp_<s>_pts_q.jsonl   one line per grid point: status (optimal / infeasible / timeout_nolb / error:*), label (cost | inf | null), y = 4 outlet mol flows,
                                                  q = (flow, conc, cf, rec), box = grid index.   (_pen_q.jsonl in penalty mode; no _q suffix with SPACE="outlet")
                   EDUnits_nocp_<s>_pts_q.progress   one line per finished chunk (resume: finished chunks are skipped; delete both files to start over)
Train / certify from it with pipeline.py: set DATA_FILE accordingly (see pipeline.py DATA_SUFFIX) and RUN_SAMPLE=False (pipeline.py only reads optimal / timeout_lb rows).

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
MODE      = "fixed"                # "fixed": fix the 4 variables at the grid point (every point is recorded)  |  "penalty": smooth extension, see below (env SAMPLE_MODE overrides)
PEN_RHO   = 1e7                    # penalty mode: weight of the scaled distance (y - grid point)/range; larger = closer to the true cost on the feasible set, steeper outside it
PEN_EPS   = 1e-3                   # penalty mode: smoothing of the distance norm (scaled units), keeps the objective differentiable at distance 0
N_PTS     = 20                     # grid points per free dimension: one int or a dict keyed by variable name (second command-line argument overrides the int)
N_PROC    = 10                     # worker processes (overridden by $SLURM_CPUS_PER_TASK)
MAX_POINTS = None                  # e.g. 2000 for a timing test (first MAX_POINTS grid points only); None = whole grid
SOLVER_NAME = "baron"              # "baron" (global solve: infeasible = proven, label = global optimum / lower bound)  |  "ipopt" (local, fast; env SAMPLE_SOLVER overrides)
BARON_EXE = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/privatemodule/baron/baron"            # None -> found on PATH   (same as pipeline.py)
BARON_LIC = "/storage/home/hcoda1/3/jyang872/r-jscott319-0/privatemodule/baron/baronlice.txt"    # None -> default
SAMPLE_TIME = 60                   # time limit (s) per point solve, BARON or IPOPT
BARON_OPTIONS = {"MaxTime": SAMPLE_TIME, "EpsR": 1e-2, "EpsA": 1e-6, "TDo": 0, "AbsConFeasTol": 1e-2}     # same as pipeline.py SAMPLE_OPTIONS: time limit (s) and gap per point
IPOPT_OPTIONS = {"tol": 1e-6, "constr_viol_tol": 1e-2, "acceptable_tol": 1e-4, "max_iter": 3000, "max_cpu_time": SAMPLE_TIME, "print_level": 0}
CHUNK     = 2000                   # grid points per task (a chunk is the unit of resume)
LOG_EVERY = 60
SCENARIO_OUTLETS = {1: [2, 3], 2: [5, 6], 3: [7, 8]}
UNIT_INLET = {1: 1, 2: 4, 3: 5}                                              # scenario -> inlet stream (= dilute + concentrate outlet)
SPACE     = "fcr"                  # sampling space: "fcr" = grid over (flow, conc, cf, rec) of the unit inlet, outlets back-calculated  |  "outlet" = old grid over the 4 outlet mol flows (env SAMPLE_SPACE overrides)
# bounds of the 4 sampling quantities = the bounds of samplingED.py (flowIn / concIn lines 131-132, CF / REC lines 149-150).  flow: inlet volumetric flow [L/d], conc: inlet N concentration [mg/L],
# cf = conc(concentrate)/conc(inlet) (mass-concentration basis, as in the model), rec = N(concentrate)/N(inlet).   Grid spacing per dimension: "log" or "lin".
QBOUNDS = {"flow": (15000.0, 250000.0), "conc": (1366.0, 15000.0), "cf": (1.05, 30.0), "rec": (0.03, 0.95)}
QSPACING = {"flow": "log", "conc": "log", "cf": "log", "rec": "lin"}
CONC_SECOND = {1: True, 2: False, 3: False}                                  # scenario -> is the SECOND outlet (N_b, W_b) the concentrate?  unit1: (2 dil, 3 conc); unit2: (5 conc, 6 dil); unit3: (7 conc, 8 dil)
RANGES = {f"mol{x}Stream{k}": ((1e-5, 10.0) if x == "N" else (1e-5, 500.0)) for k in range(1, 9) for x in "NW"}
# ====================================================================================

MODE = os.environ.get("SAMPLE_MODE", MODE); assert MODE in ("fixed", "penalty")
SPACE = os.environ.get("SAMPLE_SPACE", SPACE); assert SPACE in ("fcr", "outlet")
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
TAGF = ("pts" if MODE == "fixed" else "pen") + ("_q" if SPACE == "fcr" else "")
DATA_FILE = os.path.join(OUT_DIR, f"{INSTANCE}_{SS}_{TAGF}.jsonl"); PROG_FILE = os.path.join(OUT_DIR, f"{INSTANCE}_{SS}_{TAGF}.progress")
names = [f"mol{x}Stream{k}" for k in SCENARIO_OUTLETS[SS] for x in "NW"]
gnames = names if SPACE == "outlet" else list(QBOUNDS)                          # the 4 GRID dimensions (outlet mol flows, or flow/conc/cf/rec)
npd = [int(N_PTS[n]) if isinstance(N_PTS, dict) else int(N_PTS) for n in gnames]; assert all(k >= 2 for k in npd)
def _axis(n, k):
    if SPACE == "outlet": return np.linspace(RANGES[n][0], RANGES[n][1], k)
    lo, hi = QBOUNDS[n]; return np.geomspace(lo, hi, k) if QSPACING[n] == "log" else np.linspace(lo, hi, k)
grids = [_axis(n, k) for n, k in zip(gnames, npd)]
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
if MODE == "penalty":
    # min f_s + PEN_RHO * ( sqrt( sum_i ((y_i - p_i)/range_i)^2 + EPS^2 ) - EPS )   with the 4 variables FREE inside their original bounds;
    # the penalty is >= 0 and exactly 0 at y = p, so for a feasible p the optimum is <= f_s*(p) (global solve), and the optimum is Lipschitz in p with constant ~PEN_RHO/range
    M.p = pyo.Param(range(4), mutable=True, initialize=1.0)
    _rng = [RANGES[n][1] - RANGES[n][0] for n in names]
    M.f_obj.deactivate()
    M.f_pen = pyo.Objective(expr=SM.obj_expr[SS] + PEN_RHO * (pyo.sqrt(sum(((M.y[n] - M.p[d]) / _rng[d]) ** 2 for d, n in enumerate(names)) + PEN_EPS ** 2) - PEN_EPS), sense=pyo.minimize)
SOLVER_NAME = os.environ.get("SAMPLE_SOLVER", SOLVER_NAME); assert SOLVER_NAME in ("baron", "ipopt")
if SOLVER_NAME == "baron":
    SOLVER = pyo.SolverFactory("baron", executable=BARON_EXE) if BARON_EXE and os.path.exists(BARON_EXE) else pyo.SolverFactory("baron")      # missing path (e.g. local test) -> baron on PATH
    SOLVE_OPTIONS = dict(BARON_OPTIONS)
    if BARON_LIC and os.path.exists(BARON_LIC): SOLVE_OPTIONS["LicName"] = BARON_LIC
else:
    SOLVER = pyo.SolverFactory("ipopt"); SOLVE_OPTIONS = dict(IPOPT_OPTIONS)
if not SOLVER.available(exception_flag=False): sys.exit(f"Pyomo cannot find the {SOLVER_NAME} executable (BARON_EXE / PATH)")
# constants copied from EDUnits_nocp.py (densityH2OGM3 [g/m3], molweightN, molweightH2OgMol [g/mol])
DENS, MN, MW = 1000 * 1000, 14.0067, 18.01528


def to_outlets(Q):
    """Q: (n,4) = (flow [L/d], conc [mg/L], cf, rec) of the unit inlet -> (n,4) outlet mol flows (N_a, W_a, N_b, W_b) in the order of SCENARIO_OUTLETS[SS].
    Same stream convention as the model:  V [m3/s], c [g/m3]:  molN = V c / MN,  molW = V (DENS - c) / MW   (stream volume = mass / DENS, so volumes balance too).
      inlet:       N_in = V c / MN,  W_in = V (DENS - c) / MW
      concentrate: N_c = rec N_in,  c_c = cf c,  V_c = N_c MN / c_c,  W_c = V_c (DENS - c_c) / MW
      dilute:      N_d = N_in - N_c, W_d = W_in - W_c"""
    flow, c, cf, rec = Q.T
    V = flow / (1000 * 86400); Nin = V * c / MN; Win = V * (DENS - c) / MW
    Nc = rec * Nin; cc = cf * c; Vc = Nc * MN / cc; Wc = Vc * (DENS - cc) / MW
    Nd, Wd = Nin - Nc, Win - Wc
    return np.stack([Nd, Wd, Nc, Wc] if CONC_SECOND[SS] else [Nc, Wc, Nd, Wd], axis=1)


def _bnd(v):
    return float(v) if v is not None and np.isfinite(v) and abs(v) < 1e20 else None


def solve_point(p):
    """Solve one point.  Returns a dict: status ('optimal' | 'infeasible' | 'timeout' | 'error:*'), label, f_value, distance, y_repaired, lower_bound, upper_bound.
    fixed mode: label = f_s at the fixed point.  penalty mode: label = optimal penalised objective.  timeout: label = last lower bound (None if there is none)."""
    if MODE == "fixed":
        for n, v in zip(names, p): M.y[n].fix(float(v))
    else:
        for d, n in enumerate(names): M.p[d] = float(p[d]); M.y[n].set_value(float(p[d]))
    M.y[f"molNStream{INLET}"].set_value(float(p[0] + p[2])); M.y[f"molWStream{INLET}"].set_value(float(p[1] + p[3]))      # start the inlet at the balance value
    out = dict(status="error", label=None, f_value=None, distance=None, y_repaired=None, lower_bound=None, upper_bound=None)
    try:
        r = SOLVER.solve(M, tee=False, load_solutions=False, options=dict(SOLVE_OPTIONS)); tc = r.solver.termination_condition
        try: out["lower_bound"], out["upper_bound"] = _bnd(r.problem.lower_bound), _bnd(r.problem.upper_bound)
        except Exception: pass
        if tc in (TC.optimal, TC.globallyOptimal, TC.locallyOptimal):
            M.solutions.load_from(r); out["status"] = "optimal"; out["f_value"] = float(pyo.value(M.f_obj.expr))
            ys = [float(M.y[n].value) for n in names]; out["y_repaired"] = ys
            out["distance"] = float(np.sqrt(sum(((ys[d] - p[d]) / (RANGES[n][1] - RANGES[n][0])) ** 2 for d, n in enumerate(names))))
            out["label"] = out["f_value"] if MODE == "fixed" else float(pyo.value(M.f_pen))
        elif tc in (TC.infeasible, TC.invalidProblem): out["status"] = "infeasible"
        else: out["status"] = "timeout"; out["label"] = out["lower_bound"]
    except Exception as e: out["status"] = "error:" + type(e).__name__
    return out


def do_chunk(c):
    """Grid points c*CHUNK .. (c+1)*CHUNK-1: returns (chunk, counts, JSON lines of ALL points)."""
    a, b = c * CHUNK, min((c + 1) * CHUNK, N_TOTAL)
    ix = np.array(np.unravel_index(np.arange(a, b), npd)).T                       # (n,4) grid indices
    Q = np.stack([grids[d][ix[:, d]] for d in range(4)], axis=1)                  # grid point in the sampling space
    P = Q if SPACE == "outlet" else to_outlets(Q)                                 # (N_a, W_a, N_b, W_b) outlet mol flows handed to the solver
    counts = {}; lines = []
    for k in range(len(P)):
        t0 = time.time()
        oob = [n for d, n in enumerate(names) if not (RANGES[n][0] <= P[k, d] <= RANGES[n][1])]
        if oob:      # a fixed variable is NOT bounds-checked by the solver, so an outlet outside its original bounds is labelled here without a solve
            out = dict(status="out_of_bound", label=None, f_value=None, distance=None, y_repaired=None, lower_bound=None, upper_bound=None)
            print(f"[points] OUT OF BOUND scenario {SS} box {ix[k].tolist()}: {oob} = {[float(P[k, names.index(n)]) for n in oob]}  -> labelled infeasible (label inf), not solved", flush=True)
        else: out = solve_point(P[k])
        st = out["status"]; counts[st] = counts.get(st, 0) + 1
        # EVERY point is written.  feasible (classifier label): optimal / timeout -> True, infeasible / out_of_bound -> False, error -> None.
        # label (regression target): optimal -> cost, timeout -> last lower bound (None if none), infeasible / out_of_bound -> inf, error -> None.
        feas = True if st in ("optimal", "timeout") else False if st in ("infeasible", "out_of_bound") else None
        rec = dict(scenario=SS, vars=names, box=ix[k].tolist(), status=st, feasible=feas,
                   label=(float("inf") if feas is False else out["label"]), y={n: float(v) for n, v in zip(names, P[k])}, seconds=round(time.time() - t0, 3))
        if SPACE == "fcr": rec["q"] = {n: float(v) for n, v in zip(gnames, Q[k])}      # the sampled (flow, conc, cf, rec); y = back-calculated outlet mol flows
        if oob: rec["out_of_bound_vars"] = oob
        if st == "optimal": rec["label_type"] = ("baron_global" if SOLVER_NAME == "baron" else "ipopt_fixed_point") if MODE == "fixed" else "penalty"
        if st == "timeout": rec.update(label_type="lower_bound", lower_bound=out["lower_bound"], upper_bound=out["upper_bound"], has_incumbent=out["upper_bound"] is not None)
        if st == "optimal" and MODE == "penalty":   # y = the GRID point (NN input); f_value / distance / y_repaired describe the repaired point; distance ~ 0 means the grid point itself is feasible
            rec.update(f_value=out["f_value"], distance=out["distance"], y_repaired={n: v for n, v in zip(names, out["y_repaired"])})
        lines.append(json.dumps(rec))
    return c, counts, lines


def read_done():
    done = {}
    if os.path.exists(PROG_FILE):
        for line in open(PROG_FILE):
            try: r = json.loads(line)
            except Exception: continue
            if r.get("n_pts") != npd or r.get("chunk_size") != CHUNK or r.get("mode", "fixed") != MODE or r.get("solver", "ipopt") != SOLVER_NAME or r.get("space", "outlet") != SPACE or (SPACE == "fcr" and r.get("qb") != [list(QBOUNDS[n]) + [QSPACING[n]] for n in gnames]) or (MODE == "penalty" and r.get("rho") != [PEN_RHO, PEN_EPS]): sys.exit(f"{PROG_FILE} was written with a different grid / chunk size / mode / penalty; move or delete it and {DATA_FILE}")
            done[r["chunk"]] = r["counts"]
    return done


def main():
    done = read_done(); todo = [c for c in range(N_CHUNKS) if c not in done]
    tot = {}
    for cnt in done.values():
        for k, v in cnt.items(): tot[k] = tot.get(k, 0) + v
    print(f"[points] scenario {SS} solver {SOLVER_NAME} mode {MODE} space {SPACE}" + (f" (rho {PEN_RHO:g}, eps {PEN_EPS:g})" if MODE == "penalty" else "") + f": grid {'x'.join(map(str, npd))} = {N_ALL} points ({N_TOTAL} used), {N_CHUNKS} chunks of {CHUNK}, {len(done)} already done, {N_PROC} processes", flush=True)
    t_all = t_log = time.time(); k = 0
    fd, fp = open(DATA_FILE, "a"), open(PROG_FILE, "a")
    with mp.get_context("fork").Pool(N_PROC) as pool:
        def _stop(*a): raise SystemExit("SIGTERM received: stopping (finished chunks are saved; run again to resume)")
        signal.signal(signal.SIGTERM, _stop)
        try:
            for c, counts, lines in pool.imap_unordered(do_chunk, todo):
                for ln in lines: fd.write(ln + "\n")
                fd.flush(); os.fsync(fd.fileno())
                fp.write(json.dumps(dict(chunk=c, n_pts=npd, chunk_size=CHUNK, mode=MODE, solver=SOLVER_NAME, space=SPACE, qb=[list(QBOUNDS[n]) + [QSPACING[n]] for n in gnames] if SPACE == "fcr" else None, rho=[PEN_RHO, PEN_EPS], counts=counts)) + "\n"); fp.flush(); os.fsync(fp.fileno())
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
