"""
pyomo_to_mcpp.py
================
Walks a Pyomo constraint's expression tree and emits C++ code that builds
an equivalent MC++ (McCormick++) DAG.

MC++ reference: https://github.com/coin-or/MCpp
Supported node types:
  - NumericConstant
  - ScalarVar / _GeneralVarData
  - SumExpression / LinearExpression
  - ProductExpression
  - DivisionExpression
  - PowExpression
  - NegationExpression
  - AbsExpression
  - UnaryFunctionExpression  (exp, log, log10, sin, cos, tan, sqrt, asin, acos, atan)
  - InequalityExpression / RangedExpression / EqualityExpression  (constraint wrappers)
"""

from __future__ import annotations

import textwrap
from typing import Any

import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import (
    SumExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    NegationExpression,
    AbsExpression,
    UnaryFunctionExpression,
    LinearExpression,
    MonomialTermExpression,
)
from pyomo.core.expr.relational_expr import (
    InequalityExpression,
    EqualityExpression,
    RangedExpression,
)
from pyomo.core.expr.numvalue import NumericConstant, value as pyo_value
from pyomo.core.base.var import ScalarVar, _GeneralVarData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# MC++ intrinsics that map 1-to-1 from Pyomo's UnaryFunctionExpression names
_UNARY_MCPP = {
    "exp":   "exp",
    "log":   "log",
    "log10": "log",   # MC++ has log; log10 = log(x)/log(10) — handled below
    "sin":   "sin",
    "cos":   "cos",
    "tan":   "tan",
    "sqrt":  "sqrt",
    "asin":  "asin",
    "acos":  "acos",
    "atan":  "atan",
    "abs":   "fabs",
}


class MCppEmitter:
    """
    Traverse a Pyomo expression tree depth-first and collect:
      - variable declarations (MC++ T variables with user-supplied bounds)
      - intermediate node assignments
      - a final expression node name

    Usage
    -----
        emitter = MCppEmitter(var_bounds={"x": (-1, 1), "y": (0, 5)})
        code = emitter.emit_constraint(model.c)
        print(code)
    """

    def __init__(self, var_bounds: dict[str, tuple[float, float]] | None = None,
                 mcpp_type: str = "T"):
        """
        Parameters
        ----------
        var_bounds : dict mapping Pyomo variable name -> (lb, ub)
            If a variable is not listed, defaults to (-1e20, 1e20).
        mcpp_type : str
            The C++ template type used for MC++ variables (default "T").
        """
        self.var_bounds = var_bounds or {}
        self.T = mcpp_type
        self._reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit_constraint(self, constraint: pyo.Constraint) -> str:
        """
        Return a self-contained C++ snippet that builds the MC++ expression
        corresponding to *constraint* and prints the resulting relaxation.
        """
        self._reset()

        # --- walk the body expression ---------------------------------
        body_var = self._visit(constraint.body)

        # --- constraint sense -----------------------------------------
        lb = constraint.lower
        ub = constraint.upper
        if lb is not None:
            lb = float(pyo_value(lb))
        if ub is not None:
            ub = float(pyo_value(ub))

        return self._render_cpp(body_var, lb, ub)

    def emit_expression(self, expr) -> str:
        """Emit C++ for a bare Pyomo expression (no constraint wrapper)."""
        self._reset()
        root = self._visit(expr)
        return self._render_cpp(root, lb=None, ub=None)

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    def _reset(self):
        self._var_decls: dict[str, str] = {}   # varname -> decl line
        self._stmts: list[str] = []            # intermediate assignments
        self._node_count = 0
        self._seen_vars: dict[str, str] = {}   # pyomo var name -> cpp var name

    def _fresh(self, prefix="v") -> str:
        self._node_count += 1
        return f"{prefix}{self._node_count}"

    # ------------------------------------------------------------------
    # Tree visitor dispatcher
    # ------------------------------------------------------------------

    def _visit(self, node) -> str:
        """Recursively visit *node* and return the C++ variable name holding it."""

        # --- Leaf: numeric constant -----------------------------------
        if isinstance(node, NumericConstant) or isinstance(node, (int, float)):
            val = float(pyo_value(node))
            name = self._fresh("c")
            self._stmts.append(f"{self.T} {name} = {val};")
            return name

        # --- Leaf: Pyomo variable -------------------------------------
        if isinstance(node, (_GeneralVarData, ScalarVar)):
            return self._declare_var(node)

        # --- SumExpression / LinearExpression -------------------------
        if isinstance(node, (SumExpression, LinearExpression)):
            return self._visit_sum(node)

        # --- MonomialTermExpression (coeff * var) ---------------------
        if isinstance(node, MonomialTermExpression):
            return self._visit_monomial(node)

        # --- ProductExpression ----------------------------------------
        if isinstance(node, ProductExpression):
            return self._visit_nary(node, "*", "prod")

        # --- DivisionExpression ---------------------------------------
        if isinstance(node, DivisionExpression):
            args = list(node.args)
            left  = self._visit(args[0])
            right = self._visit(args[1])
            name  = self._fresh("div")
            self._stmts.append(f"{self.T} {name} = {left} / {right};")
            return name

        # --- PowExpression --------------------------------------------
        if isinstance(node, PowExpression):
            args = list(node.args)
            base  = self._visit(args[0])
            exp_  = self._visit(args[1])
            name  = self._fresh("pow")
            self._stmts.append(f"{self.T} {name} = pow({base}, {exp_});")
            return name

        # --- NegationExpression ---------------------------------------
        if isinstance(node, NegationExpression):
            child = self._visit(list(node.args)[0])
            name  = self._fresh("neg")
            self._stmts.append(f"{self.T} {name} = -({child});")
            return name

        # --- AbsExpression --------------------------------------------
        if isinstance(node, AbsExpression):
            child = self._visit(list(node.args)[0])
            name  = self._fresh("abs")
            self._stmts.append(f"{self.T} {name} = fabs({child});")
            return name

        # --- UnaryFunctionExpression ----------------------------------
        if isinstance(node, UnaryFunctionExpression):
            return self._visit_unary(node)

        # --- Constraint wrapper types (shouldn't appear in .body, but
        #     handle gracefully if someone passes .expr instead) -------
        if isinstance(node, (InequalityExpression, EqualityExpression,
                              RangedExpression)):
            return self._visit(node.args[0])  # recurse into body

        raise NotImplementedError(
            f"Unsupported Pyomo node type: {type(node).__name__}  node={node}"
        )

    # ------------------------------------------------------------------
    # Specialised visitors
    # ------------------------------------------------------------------

    def _declare_var(self, var_node) -> str:
        """Emit a MC++ variable declaration with interval bounds."""
        pyomo_name = var_node.name          # e.g. "model.x"
        short_name = var_node.local_name    # e.g. "x"

        if pyomo_name in self._seen_vars:
            return self._seen_vars[pyomo_name]

        cpp_name = short_name.replace("[", "_").replace("]", "_").replace(",", "_")
        # Make unique if name collision
        while cpp_name in self._seen_vars.values():
            cpp_name += "_"

        lb_py = var_node.lb
        ub_py = var_node.ub
        lb = float(pyo_value(lb_py)) if lb_py is not None else self.var_bounds.get(short_name, (-1e20, 1e20))[0]
        ub = float(pyo_value(ub_py)) if ub_py is not None else self.var_bounds.get(short_name, (-1e20, 1e20))[1]

        decl = (f"{self.T} {cpp_name}"
                f"( I({lb}, {ub}) );  // Pyomo var: {pyomo_name}")
        self._var_decls[pyomo_name] = decl
        self._seen_vars[pyomo_name] = cpp_name
        return cpp_name

    def _visit_sum(self, node) -> str:
        args = list(node.args)
        if not args:
            name = self._fresh("zero")
            self._stmts.append(f"{self.T} {name} = 0.0;")
            return name

        visited = [self._visit(a) for a in args]
        # Build as a running sum to keep the C++ readable
        name = self._fresh("sum")
        rhs = " + ".join(visited)
        self._stmts.append(f"{self.T} {name} = {rhs};")
        return name

    def _visit_monomial(self, node) -> str:
        args = list(node.args)          # (coeff, var)
        coeff = self._visit(args[0])
        var   = self._visit(args[1])
        name  = self._fresh("mono")
        self._stmts.append(f"{self.T} {name} = {coeff} * {var};")
        return name

    def _visit_nary(self, node, op: str, prefix: str) -> str:
        args = list(node.args)
        visited = [self._visit(a) for a in args]
        name = self._fresh(prefix)
        rhs  = f" {op} ".join(visited)
        self._stmts.append(f"{self.T} {name} = {rhs};")
        return name

    def _visit_unary(self, node: UnaryFunctionExpression) -> str:
        fname = node.getname()
        child = self._visit(list(node.args)[0])
        name  = self._fresh(fname[:3])

        if fname == "log10":
            # log10(x) = log(x) / log(10)
            tmp = self._fresh("log")
            self._stmts.append(f"{self.T} {tmp} = log({child});")
            log10val = self._fresh("c")
            self._stmts.append(f"{self.T} {log10val} = {import_log10()};")
            self._stmts.append(f"{self.T} {name} = {tmp} / {log10val};")
        else:
            mcpp_fn = _UNARY_MCPP.get(fname)
            if mcpp_fn is None:
                raise NotImplementedError(
                    f"Unary function '{fname}' not mapped to MC++.")
            self._stmts.append(f"{self.T} {name} = {mcpp_fn}({child});")

        return name

    # ------------------------------------------------------------------
    # C++ code renderer
    # ------------------------------------------------------------------

    def _render_cpp(self, body_var: str, lb, ub) -> str:
        lines: list[str] = []

        lines.append("// -------------------------------------------------------")
        lines.append("// MC++ expression generated from Pyomo constraint")
        lines.append("// Requires: mc++/mcpp.hpp  (MC++ library)")
        lines.append("// -------------------------------------------------------")
        lines.append("")
        lines.append("#include <mc++/mcpp.hpp>")
        lines.append("#include <interval/interval.hpp>  // or your IA header")
        lines.append("")
        lines.append("using namespace mc;")
        lines.append("using I = Interval;  // or your interval type")
        lines.append(f"using {self.T} = McCormick<I>;")
        lines.append("")
        lines.append("void evaluate_constraint() {")
        lines.append("")

        if self._var_decls:
            lines.append("  // --- Decision variables (with interval bounds) ---")
            for decl in self._var_decls.values():
                lines.append(f"  {decl}")
            lines.append("")

        lines.append("  // --- Expression tree ---")
        for stmt in self._stmts:
            lines.append(f"  {stmt}")
        lines.append("")

        lines.append(f"  // --- Result: body of constraint ---")
        lines.append(f"  {self.T} body = {body_var};")
        lines.append("")

        if lb is not None or ub is not None:
            lines.append("  // --- Constraint bounds ---")
            if lb is not None:
                lines.append(f"  double lb = {lb};")
            if ub is not None:
                lines.append(f"  double ub = {ub};")
            lines.append("")

        lines.append("  // --- Inspect the relaxation ---")
        lines.append("  std::cout << \"Interval:   \" << body.I()  << std::endl;")
        lines.append("  std::cout << \"CV (convex):\" << body.cv() << std::endl;")
        lines.append("  std::cout << \"CC (concave):\"<< body.cc() << std::endl;")
        lines.append("}")
        lines.append("")

        return "\n".join(lines)


def import_log10():
    import math
    return str(math.log(10))


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------

def constraint_to_mcpp(constraint: pyo.Constraint,
                       var_bounds: dict[str, tuple[float, float]] | None = None,
                       mcpp_type: str = "T") -> str:
    """
    Convert a Pyomo Constraint to an MC++ C++ code snippet.

    Parameters
    ----------
    constraint  : pyo.Constraint   — a concrete (scalar) Pyomo constraint
    var_bounds  : dict, optional   — fallback bounds {var_name: (lb, ub)}
                                     used only when the Pyomo var has no bounds
    mcpp_type   : str              — C++ type alias (default "T")

    Returns
    -------
    str — C++ source code
    """
    emitter = MCppEmitter(var_bounds=var_bounds, mcpp_type=mcpp_type)
    return emitter.emit_constraint(constraint)


def expr_to_mcpp(expr,
                 var_bounds: dict[str, tuple[float, float]] | None = None,
                 mcpp_type: str = "T") -> str:
    """Convert a bare Pyomo expression (not wrapped in a Constraint)."""
    emitter = MCppEmitter(var_bounds=var_bounds, mcpp_type=mcpp_type)
    return emitter.emit_expression(expr)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-2, 2))
    m.y = pyo.Var(bounds=(0, 4))
    m.z = pyo.Var(bounds=(0.1, 10))

    # ---- Example 1: polynomial constraint  x^2 + 2*x*y - y <= 5 ----
    m.c1 = pyo.Constraint(expr=m.x**2 + 2*m.x*m.y - m.y <= 5)
    print("=" * 60)
    print("EXAMPLE 1 — polynomial:  x^2 + 2*x*y - y <= 5")
    print("=" * 60)
    print(constraint_to_mcpp(m.c1))

    # ---- Example 2: transcendental  exp(x) + log(z) == y  ----------
    m.c2 = pyo.Constraint(expr=pyo.exp(m.x) + pyo.log(m.z) == m.y)
    print("=" * 60)
    print("EXAMPLE 2 — transcendental:  exp(x) + log(z) == y")
    print("=" * 60)
    print(constraint_to_mcpp(m.c2))

    # ---- Example 3: ranged  -1 <= sin(x)*cos(y) <= 1 ---------------
    m.c3 = pyo.Constraint(expr=pyo.inequality(-1, pyo.sin(m.x)*pyo.cos(m.y), 1))
    print("=" * 60)
    print("EXAMPLE 3 — ranged trig:  -1 <= sin(x)*cos(y) <= 1")
    print("=" * 60)
    print(constraint_to_mcpp(m.c3))

    # ---- Example 4: division + abs  abs(x/z) <= 3 ------------------
    m.c4 = pyo.Constraint(expr=abs(m.x / m.z) <= 3)
    print("=" * 60)
    print("EXAMPLE 4 — abs+div:  abs(x/z) <= 3")
    print("=" * 60)
    print(constraint_to_mcpp(m.c4))