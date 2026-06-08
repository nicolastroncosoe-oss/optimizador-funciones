from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import csv
import io
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# =============================================================================
# Configuración numérica
# =============================================================================

SP_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

LOG_FLOOR = 1e-16
ALPHA_EPS = 1e-14
MAX_LINESEARCH_ITERS = 70
MAX_ZOOM_ITERS = 70
MAX_ALPHA_WOLFE = 20.0
MAX_GRID_POINTS = 130

DIVERGENCE_X_NORM = 1e12
DIVERGENCE_F_ABS = 1e200
DIVERGENCE_G_NORM = 1e150

SMOOTHNESS_ATOMS = (sp.Abs, sp.Max, sp.Min, sp.Piecewise)

ALLOWED_FUNCS: Dict[str, object] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
}

METHODS = ["Gradiente", "Gradiente Conjugado", "Newton"]
ALPHA_MODES = ["Automático Wolfe", "Alpha fijo"]

EXAMPLES = {
    "Cuadrática simple 2D": {
        "f": "x1^2 + x2^2",
        "n": 2,
        "x0": "2, 2",
        "method": "Newton",
        "tol": 1e-8,
        "max_iter": 100,
    },
    "Rosenbrock": {
        "f": "100*(x2 - x1^2)^2 + (1 - x1)^2",
        "n": 2,
        "x0": "-1.2, 1",
        "method": "Newton",
        "tol": 1e-8,
        "max_iter": 300,
    },
    "Himmelblau": {
        "f": "(x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2",
        "n": 2,
        "x0": "-3, 3",
        "method": "Newton",
        "tol": 1e-8,
        "max_iter": 300,
    },
    "Oscilatoria 2D para superficie": {
        "f": "0.08*(x1^2 + x2^2) + sin(x1)*cos(x2) + 0.15*sin(3*x1 + x2) + 0.05*(x1 - x2)^2",
        "n": 2,
        "x0": "2.5, -2",
        "method": "Newton",
        "tol": 1e-8,
        "max_iter": 300,
    },
    "Cinco variables": {
        "f": "x1^2 + 2*x2^2 + 3*x3^2 + 4*x4^2 + 5*x5^2",
        "n": 5,
        "x0": "1, 1, 1, 1, 1",
        "method": "Newton",
        "tol": 1e-8,
        "max_iter": 100,
    },
    "Diez variables": {
        "f": "x1^2 + 2*x2^2 + 3*x3^2 + 4*x4^2 + 5*x5^2 + 6*x6^2 + 7*x7^2 + 8*x8^2 + 9*x9^2 + 10*x10^2",
        "n": 10,
        "x0": "1,1,1,1,1,1,1,1,1,1",
        "method": "Newton",
        "tol": 1e-8,
        "max_iter": 100,
    },
}

DEFAULT_EXAMPLE = "Cuadrática simple 2D"


# =============================================================================
# Estado de la interfaz
# =============================================================================

def load_example_into_state(example_name: str) -> None:
    ex = EXAMPLES[example_name]
    st.session_state["example_selected"] = example_name
    st.session_state["n_vars"] = int(ex["n"])
    st.session_state["funcion"] = str(ex["f"])
    st.session_state["metodo"] = str(ex["method"])
    st.session_state["punto_inicial"] = str(ex["x0"])
    st.session_state["max_iter"] = int(ex["max_iter"])
    st.session_state["tolerancia"] = float(ex["tol"])


def ensure_default_state() -> None:
    if "example_selected" not in st.session_state:
        load_example_into_state(DEFAULT_EXAMPLE)

    st.session_state.setdefault("c1_input", 1e-4)
    st.session_state.setdefault("c2_input", 0.9)
    st.session_state.setdefault("modo_alpha", "Automático Wolfe")
    st.session_state.setdefault("alpha_fijo", 0.1)


def update_c2_recommendation() -> None:
    metodo = st.session_state.get("metodo", "Newton")
    st.session_state["c2_input"] = 0.4 if metodo == "Gradiente Conjugado" else 0.9


# =============================================================================
# Estructuras de datos
# =============================================================================

@dataclass
class Objective:
    f_sym: sp.Expr
    variables: Tuple[sp.Symbol, ...]
    grad_sym: List[sp.Expr]
    hess_sym: List[List[sp.Expr]]
    f_raw: Callable
    grad_raw: Sequence[Callable]
    hess_raw: Sequence[Sequence[Callable]]

    def f(self, x: np.ndarray) -> float:
        return safe_eval_f(self.f_raw, np.asarray(x, dtype=float))

    def grad(self, x: np.ndarray) -> np.ndarray:
        return safe_eval_grad(self.grad_raw, np.asarray(x, dtype=float))

    def hess(self, x: np.ndarray) -> np.ndarray:
        return safe_eval_hess(self.hess_raw, np.asarray(x, dtype=float))


@dataclass
class AlphaSearchInfo:
    alpha_inicial: Optional[float] = None
    alpha_aceptado: Optional[float] = None
    wolfe_iters: int = 0
    zoom_iters: int = 0
    total_internas: int = 0
    modo_alpha: str = "—"


@dataclass
class IterRecord:
    iteracion: int
    x: np.ndarray
    f: float
    grad_norm: float
    alpha: Optional[float] = None
    direccion: str = "—"
    wolfe_armijo: Optional[bool] = None
    wolfe_curvatura: Optional[bool] = None
    metodo_paso: str = "—"
    beta: Optional[float] = None
    alpha_inicial: Optional[float] = None
    wolfe_iters: int = 0
    zoom_iters: int = 0
    total_internas: int = 0
    modo_alpha: str = "—"


@dataclass
class OptimizationResult:
    records: List[IterRecord]
    stop_reason: str
    converged: bool
    hessian_class: str
    hessian_message: str
    eigvals: Optional[np.ndarray]
    best_index: int


# =============================================================================
# Utilidades numéricas
# =============================================================================

def fmt(v, fmt_spec: str = ".4e") -> str:
    if v is None:
        return "—"

    try:
        vf = float(v)
        if not np.isfinite(vf):
            return "no finito"
        return f"{vf:{fmt_spec}}"
    except Exception:
        return str(v)


def finite_float(value) -> float:
    arr = np.asarray(value)

    if arr.size != 1:
        arr = np.squeeze(arr)

    if np.iscomplexobj(arr):
        if np.max(np.abs(np.imag(arr))) > 1e-10:
            raise ValueError("La evaluación produjo un valor complejo.")
        arr = np.real(arr)

    out = float(np.asarray(arr, dtype=float))

    if not np.isfinite(out):
        raise ValueError("La evaluación produjo NaN o infinito.")

    return out


def safe_eval_f(f_raw: Callable, x: np.ndarray) -> float:
    return finite_float(f_raw(*x))


def safe_eval_grad(grad_raw: Sequence[Callable], x: np.ndarray) -> np.ndarray:
    vals = np.array([finite_float(g(*x)) for g in grad_raw], dtype=float)

    if not np.all(np.isfinite(vals)):
        raise ValueError("El gradiente contiene NaN o infinito.")

    return vals


def safe_eval_hess(hess_raw: Sequence[Sequence[Callable]], x: np.ndarray) -> np.ndarray:
    H = np.array([[finite_float(h(*x)) for h in row] for row in hess_raw], dtype=float)

    if not np.all(np.isfinite(H)):
        raise ValueError("La Hessiana contiene NaN o infinito.")

    return H


def vector_to_string(x: np.ndarray, precision: int = 8) -> str:
    return str(np.round(np.asarray(x, dtype=float), precision).tolist())


def safe_log_values(values: Sequence[float], floor: float = LOG_FLOOR) -> List[Optional[float]]:
    out: List[Optional[float]] = []

    for v in values:
        if v is None or not np.isfinite(v):
            out.append(None)
        else:
            out.append(max(float(v), floor))

    return out


def is_unstable_state(x: np.ndarray, fx: float, grad_norm: float) -> bool:
    return (
        not np.all(np.isfinite(x))
        or not np.isfinite(fx)
        or not np.isfinite(grad_norm)
        or np.linalg.norm(x) > DIVERGENCE_X_NORM
        or abs(fx) > DIVERGENCE_F_ABS
        or grad_norm > DIVERGENCE_G_NORM
    )


def robust_range(values: np.ndarray, min_pad: float = 1.0, max_span: float = 1e6) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return -min_pad, min_pad

    lo, hi = np.percentile(vals, [2, 98]) if vals.size >= 5 else (np.min(vals), np.max(vals))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return -min_pad, min_pad

    if abs(hi - lo) < 1e-12:
        center = 0.5 * (hi + lo)
        return center - min_pad, center + min_pad

    center = 0.5 * (hi + lo)
    span = min(abs(hi - lo), max_span)
    pad = max(min_pad, 0.25 * span)

    return center - 0.5 * span - pad, center + 0.5 * span + pad


# =============================================================================
# Parser simbólico
# =============================================================================

def detectar_variables_escritas(funcion: str) -> List[int]:
    tokens = re.findall(r"\bx(\d+)\b", funcion)
    return sorted({int(t) for t in tokens})


def parse_objective(funcion: str, n: int) -> Objective:
    if not funcion.strip():
        raise ValueError("Ingresa una función objetivo.")

    indices_usados = detectar_variables_escritas(funcion)

    if indices_usados:
        max_indice = max(indices_usados)

        if max_indice > n:
            variables_fuera = [f"x{i}" for i in indices_usados if i > n]
            raise ValueError(
                f"La función usa {variables_fuera}, pero seleccionaste {n} variable(s). "
                f"Cambia el número de variables a {max_indice} o elimina esas variables."
            )

    variables = sp.symbols(f"x1:{n + 1}")

    if n == 1 and not isinstance(variables, tuple):
        variables = (variables,)

    variables = tuple(variables)

    parser_symbols = sp.symbols("x1:51")
    local_dict = {str(v): v for v in parser_symbols}
    local_dict.update(ALLOWED_FUNCS)

    try:
        f_sym = parse_expr(
            funcion,
            transformations=SP_TRANSFORMS,
            local_dict=local_dict,
            evaluate=True,
        )
    except Exception as e:
        raise ValueError(
            "No se pudo interpretar la función. Ejemplos válidos: "
            "x1^2 + x2^2, 3*x3^2, sin(x1), exp(x1), x1*x2. "
            f"Detalle: {e}"
        )

    extras = f_sym.free_symbols - set(variables)

    if extras:
        extra_names = sorted(str(e) for e in extras)
        product_hint = ""

        if re.search(r"x\d+x\d+", funcion.replace(" ", "")):
            product_hint = " Para multiplicar variables escribe x1*x2 o x1 x2; no escribas x1x2."

        raise ValueError(
            f"La función usa variables no permitidas: {extra_names}. "
            f"Usa solamente x1,...,x{n}." + product_hint
        )

    if any(f_sym.has(atom) for atom in SMOOTHNESS_ATOMS):
        raise ValueError(
            "La función contiene elementos no suaves como Abs, Max, Min o Piecewise. "
            "Usa una función diferenciable para estos métodos."
        )

    grad_sym = [sp.diff(f_sym, v) for v in variables]
    hess_sym = [[sp.diff(g, v) for v in variables] for g in grad_sym]

    try:
        f_raw = sp.lambdify(variables, f_sym, "numpy")
        grad_raw = [sp.lambdify(variables, g, "numpy") for g in grad_sym]
        hess_raw = [[sp.lambdify(variables, h, "numpy") for h in row] for row in hess_sym]
    except Exception as e:
        raise ValueError(f"No se pudieron construir las funciones numéricas: {e}")

    return Objective(f_sym, variables, grad_sym, hess_sym, f_raw, grad_raw, hess_raw)


def is_affine_objective(obj: Objective) -> bool:
    try:
        return all(sp.simplify(h) == 0 for row in obj.hess_sym for h in row)
    except Exception:
        return False


# =============================================================================
# Clasificación por Hessiana
# =============================================================================

def classify_hessian(
    H: Optional[np.ndarray],
    grad_norm: Optional[float],
    tol: float,
    eig_tol: float = 1e-8,
) -> Tuple[str, str, Optional[np.ndarray]]:
    stationary_threshold = max(10.0 * float(tol), 1e-8)

    if grad_norm is None or (not np.isfinite(grad_norm)) or float(grad_norm) > stationary_threshold:
        return (
            "no_estacionario",
            "El punto final no cumple la condición de estacionariedad; la clasificación por Hessiana no se aplica.",
            None,
        )

    if H is None:
        return "no_clasificado", "No se pudo calcular la Hessiana final.", None

    try:
        Hs = 0.5 * (H + H.T)
        eigvals = np.linalg.eigvalsh(Hs)
    except Exception:
        return "no_clasificado", "No se pudieron calcular los valores propios de la Hessiana.", None

    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))

    if min_eig > eig_tol:
        return "minimo_local", "Hessiana definida positiva: candidato fuerte a mínimo local.", eigvals

    if max_eig < -eig_tol:
        return "maximo_local", "Hessiana definida negativa en un punto estacionario: candidato a máximo local.", eigvals

    if min_eig < -eig_tol and max_eig > eig_tol:
        return "silla", "Hessiana indefinida en un punto estacionario: punto silla.", eigvals

    return (
        "degenerado",
        "La Hessiana es semidefinida o casi singular. El gradiente cumple la tolerancia, pero la prueba de segundo orden no es concluyente.",
        eigvals,
    )


# =============================================================================
# Wolfe con registro de iteraciones internas
# =============================================================================

def make_wolfe_search(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    c1: float,
    c2: float,
):
    def valid_phi(x: np.ndarray, p: np.ndarray, alpha: float) -> Tuple[bool, float]:
        try:
            val = f(x + alpha * p)
            return np.isfinite(val), val
        except Exception:
            return False, float("nan")

    def valid_dphi(x: np.ndarray, p: np.ndarray, alpha: float) -> Tuple[bool, float]:
        try:
            val = float(np.dot(grad(x + alpha * p), p))
            return np.isfinite(val), val
        except Exception:
            return False, float("nan")

    def zoom(
        alpha_lo: float,
        alpha_hi: float,
        x: np.ndarray,
        p: np.ndarray,
        phi0: float,
        dphi0: float,
    ) -> Tuple[float, int]:
        ok_lo, phi_lo = valid_phi(x, p, alpha_lo)

        if not ok_lo:
            return 0.0, 0

        zoom_iters = 0

        for _ in range(MAX_ZOOM_ITERS):
            zoom_iters += 1

            if abs(alpha_hi - alpha_lo) < 1e-13:
                break

            alpha = 0.5 * (alpha_lo + alpha_hi)
            ok_a, phi_a = valid_phi(x, p, alpha)

            if not ok_a:
                alpha_hi = alpha
                continue

            if phi_a > phi0 + c1 * alpha * dphi0 or phi_a >= phi_lo:
                alpha_hi = alpha
            else:
                ok_da, dphi_a = valid_dphi(x, p, alpha)

                if not ok_da:
                    alpha_hi = alpha
                    continue

                if abs(dphi_a) <= -c2 * dphi0:
                    return alpha, zoom_iters

                if dphi_a * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo

                alpha_lo = alpha
                phi_lo = phi_a

        return max(0.0, 0.5 * (alpha_lo + alpha_hi)), zoom_iters

    def wolfe_search(x: np.ndarray, p: np.ndarray) -> Tuple[float, AlphaSearchInfo]:
        alpha_inicial = 1.0
        info = AlphaSearchInfo(
            alpha_inicial=alpha_inicial,
            alpha_aceptado=None,
            wolfe_iters=0,
            zoom_iters=0,
            total_internas=0,
            modo_alpha="Automático Wolfe",
        )

        try:
            phi0 = f(x)
            dphi0 = float(np.dot(grad(x), p))
        except Exception:
            info.alpha_aceptado = 0.0
            return 0.0, info

        if not np.isfinite(phi0) or not np.isfinite(dphi0) or dphi0 >= 0:
            info.alpha_aceptado = 0.0
            return 0.0, info

        alpha_prev = 0.0
        phi_prev = phi0
        alpha = alpha_inicial

        for i in range(MAX_LINESEARCH_ITERS):
            info.wolfe_iters += 1
            ok_a, phi_a = valid_phi(x, p, alpha)

            if not ok_a:
                alpha_final, z_iters = zoom(alpha_prev, alpha, x, p, phi0, dphi0)
                info.zoom_iters += z_iters
                info.total_internas = info.wolfe_iters + info.zoom_iters
                info.alpha_aceptado = alpha_final
                return alpha_final, info

            if phi_a > phi0 + c1 * alpha * dphi0 or (i > 0 and phi_a >= phi_prev):
                alpha_final, z_iters = zoom(alpha_prev, alpha, x, p, phi0, dphi0)
                info.zoom_iters += z_iters
                info.total_internas = info.wolfe_iters + info.zoom_iters
                info.alpha_aceptado = alpha_final
                return alpha_final, info

            ok_da, dphi_a = valid_dphi(x, p, alpha)

            if not ok_da:
                alpha_final, z_iters = zoom(alpha_prev, alpha, x, p, phi0, dphi0)
                info.zoom_iters += z_iters
                info.total_internas = info.wolfe_iters + info.zoom_iters
                info.alpha_aceptado = alpha_final
                return alpha_final, info

            if abs(dphi_a) <= -c2 * dphi0:
                info.total_internas = info.wolfe_iters + info.zoom_iters
                info.alpha_aceptado = alpha
                return alpha, info

            if dphi_a >= 0:
                alpha_final, z_iters = zoom(alpha, alpha_prev, x, p, phi0, dphi0)
                info.zoom_iters += z_iters
                info.total_internas = info.wolfe_iters + info.zoom_iters
                info.alpha_aceptado = alpha_final
                return alpha_final, info

            alpha_prev = alpha
            phi_prev = phi_a
            alpha = min(2.0 * alpha, MAX_ALPHA_WOLFE)

        info.total_internas = info.wolfe_iters + info.zoom_iters
        info.alpha_aceptado = 0.0
        return 0.0, info

    def verificar_wolfe(x: np.ndarray, p: np.ndarray, alpha: float) -> Tuple[bool, bool]:
        try:
            phi0 = f(x)
            dphi0 = float(np.dot(grad(x), p))
            phi_a = f(x + alpha * p)
            dphi_a = float(np.dot(grad(x + alpha * p), p))

            if not all(np.isfinite(v) for v in [phi0, dphi0, phi_a, dphi_a]):
                return False, False

            return (
                bool(phi_a <= phi0 + c1 * alpha * dphi0),
                bool(abs(dphi_a) <= c2 * abs(dphi0)),
            )
        except Exception:
            return False, False

    return wolfe_search, verificar_wolfe


# =============================================================================
# Optimización
# =============================================================================

def optimize_objective(
    obj: Objective,
    x0: np.ndarray,
    method: str,
    max_iter: int,
    tol: float,
    c1: float,
    c2: float,
    modo_alpha: str = "Automático Wolfe",
    alpha_fijo: float = 0.1,
) -> OptimizationResult:
    f, grad, hess = obj.f, obj.grad, obj.hess
    wolfe_search, verificar_wolfe = make_wolfe_search(f, grad, c1, c2)

    def elegir_alpha(x_actual: np.ndarray, direccion_p: np.ndarray) -> Tuple[float, AlphaSearchInfo]:
        if modo_alpha == "Alpha fijo":
            info = AlphaSearchInfo(
                alpha_inicial=float(alpha_fijo),
                alpha_aceptado=float(alpha_fijo),
                wolfe_iters=0,
                zoom_iters=0,
                total_internas=0,
                modo_alpha="Alpha fijo",
            )
            return float(alpha_fijo), info

        return wolfe_search(x_actual, direccion_p)

    records: List[IterRecord] = []
    x = np.asarray(x0, dtype=float).copy()
    stop_reason = "máximo de iteraciones"

    def add_record(
        k: int,
        x_val: np.ndarray,
        alpha=None,
        direccion="—",
        w1=None,
        w2=None,
        metodo_paso="—",
        beta=None,
        alpha_info: Optional[AlphaSearchInfo] = None,
    ):
        fx = f(x_val)
        gx = grad(x_val)
        gnorm = float(np.linalg.norm(gx))

        if is_unstable_state(x_val, fx, gnorm):
            raise FloatingPointError("estado numérico inestable")

        if alpha_info is None:
            alpha_info = AlphaSearchInfo()

        records.append(
            IterRecord(
                iteracion=k,
                x=x_val.copy(),
                f=float(fx),
                grad_norm=gnorm,
                alpha=alpha,
                direccion=direccion,
                wolfe_armijo=w1,
                wolfe_curvatura=w2,
                metodo_paso=metodo_paso,
                beta=beta,
                alpha_inicial=alpha_info.alpha_inicial,
                wolfe_iters=alpha_info.wolfe_iters,
                zoom_iters=alpha_info.zoom_iters,
                total_internas=alpha_info.total_internas,
                modo_alpha=alpha_info.modo_alpha,
            )
        )

    add_record(0, x)

    if records[-1].grad_norm < tol:
        stop_reason = "el punto inicial cumple la tolerancia"

    elif is_affine_objective(obj):
        stop_reason = "función afín: el gradiente es constante y no existe punto estacionario salvo el caso constante"

    elif method == "Gradiente":
        for k in range(1, max_iter + 1):
            g = grad(x)

            if float(np.linalg.norm(g)) < tol:
                stop_reason = "tolerancia alcanzada"
                break

            p = -g
            alpha, alpha_info = elegir_alpha(x, p)

            if alpha <= ALPHA_EPS:
                stop_reason = "paso mínimo alcanzado por la búsqueda de línea"
                break

            w1, w2 = verificar_wolfe(x, p, alpha)
            x_new = x + alpha * p

            try:
                add_record(k, x_new, alpha, "-∇f", w1, w2, "Gradiente", alpha_info=alpha_info)
            except Exception:
                stop_reason = "posible divergencia, salida del dominio o problema no acotado inferiormente"
                break

            x = x_new

            if records[-1].grad_norm < tol:
                stop_reason = "tolerancia alcanzada"
                break

    elif method == "Gradiente Conjugado":
        g = grad(x)
        p = -g
        n = len(x)

        for k in range(1, max_iter + 1):
            if float(np.linalg.norm(g)) < tol:
                stop_reason = "tolerancia alcanzada"
                break

            restart = (k - 1) % max(n, 1) == 0

            if restart or np.dot(p, g) >= 0 or not np.all(np.isfinite(p)):
                p = -g

            alpha, alpha_info = elegir_alpha(x, p)
            step_name = "CG" if not restart else "reinicio periódico"

            if alpha <= ALPHA_EPS:
                p = -g
                alpha, alpha_info = elegir_alpha(x, p)
                step_name = "reinicio a -∇f"

                if alpha <= ALPHA_EPS:
                    stop_reason = "paso mínimo alcanzado por la búsqueda de línea"
                    break

            w1, w2 = verificar_wolfe(x, p, alpha)
            x_new = x + alpha * p

            try:
                g_new = grad(x_new)
                denom = float(np.dot(g, g))
                beta = max(0.0, float(np.dot(g_new, g_new - g)) / denom) if denom > 1e-24 else 0.0

                add_record(
                    k,
                    x_new,
                    alpha,
                    "CG" if step_name == "CG" else "-∇f",
                    w1,
                    w2,
                    step_name,
                    beta,
                    alpha_info=alpha_info,
                )
            except Exception:
                stop_reason = "posible divergencia, salida del dominio o problema no acotado inferiormente"
                break

            p = -g_new + beta * p
            x, g = x_new, g_new

            if records[-1].grad_norm < tol:
                stop_reason = "tolerancia alcanzada"
                break

    elif method == "Newton":
        for k in range(1, max_iter + 1):
            g = grad(x)

            if float(np.linalg.norm(g)) < tol:
                stop_reason = "tolerancia alcanzada"
                break

            direccion = "Newton"
            step_name = "Newton amortiguado"

            try:
                H = hess(x)
                Hs = 0.5 * (H + H.T)
                np.linalg.cholesky(Hs)
                p = -np.linalg.solve(Hs, g)

                if not np.all(np.isfinite(p)) or np.dot(p, g) >= 0:
                    raise np.linalg.LinAlgError("dirección no descendente")

            except Exception:
                p = -g
                direccion = "-∇f"
                step_name = "fallback a gradiente"

            alpha, alpha_info = elegir_alpha(x, p)

            if alpha <= ALPHA_EPS and direccion != "-∇f":
                p = -g
                alpha, alpha_info = elegir_alpha(x, p)
                direccion = "-∇f"
                step_name = "fallback a gradiente"

            if alpha <= ALPHA_EPS:
                stop_reason = "paso mínimo alcanzado por la búsqueda de línea"
                break

            w1, w2 = verificar_wolfe(x, p, alpha)
            x_new = x + alpha * p

            try:
                add_record(k, x_new, alpha, direccion, w1, w2, step_name, alpha_info=alpha_info)
            except Exception:
                stop_reason = "posible divergencia, salida del dominio o problema no acotado inferiormente"
                break

            x = x_new

            if records[-1].grad_norm < tol:
                stop_reason = "tolerancia alcanzada"
                break

    else:
        raise ValueError(f"Método no reconocido: {method}")

    final = records[-1]
    converged = bool(np.isfinite(final.grad_norm) and final.grad_norm < tol)

    try:
        H_final = hess(final.x)
    except Exception:
        H_final = None

    if converged:
        hclass, hmsg, eigvals = classify_hessian(H_final, final.grad_norm, tol)
    else:
        hclass, hmsg, eigvals = (
            "no_estacionario",
            "La clasificación por Hessiana se reserva para puntos que alcanzan la tolerancia del gradiente.",
            None,
        )

    finite_f = np.array([r.f if np.isfinite(r.f) else np.inf for r in records], dtype=float)
    best_index = int(np.argmin(finite_f)) if finite_f.size else 0

    return OptimizationResult(
        records=records,
        stop_reason=stop_reason,
        converged=converged,
        hessian_class=hclass,
        hessian_message=hmsg,
        eigvals=eigvals,
        best_index=best_index,
    )


# =============================================================================
# Gráficos
# =============================================================================

def plot_convergence(records: List[IterRecord], tol: float):
    st.subheader("Convergencia")

    iters = [r.iteracion for r in records]
    grad_norms = [r.grad_norm for r in records]
    f_vals = [r.f for r in records]
    use_markers = len(records) <= 100

    tab_grad_log, tab_grad_lin, tab_fx = st.tabs(
        ["📉 ‖∇f‖ log", "📈 ‖∇f‖ lineal", "🎯 Valor objetivo f(x)"]
    )

    def grad_fig(y_data, log_scale: bool):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=iters,
                y=y_data,
                mode="lines+markers" if use_markers else "lines",
                name="‖∇f‖",
                customdata=np.column_stack([f_vals]),
                hovertemplate=(
                    "Iteración %{x}<br>"
                    "‖∇f‖=%{y:.6e}<br>"
                    "f(x)=%{customdata[0]:.8e}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_hline(
            y=tol,
            line_dash="dash",
            annotation_text=f"Tolerancia {tol:.1e}",
            annotation_position="bottom right",
        )

        fig.update_layout(
            xaxis_title="Iteración k",
            yaxis_title="‖∇f(xₖ)‖",
            yaxis_type="log" if log_scale else "linear",
            height=430,
            hovermode="closest",
            margin=dict(l=10, r=10, t=35, b=10),
        )

        return fig

    with tab_grad_log:
        st.plotly_chart(grad_fig(safe_log_values(grad_norms), True), use_container_width=True)
        st.caption("En escala logarítmica, los ceros numéricos se representan con un piso de 1e-16.")

    with tab_grad_lin:
        st.plotly_chart(grad_fig(grad_norms, False), use_container_width=True)

    with tab_fx:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=iters,
                y=f_vals,
                mode="lines+markers" if use_markers else "lines",
                name="f(xₖ)",
                customdata=np.column_stack([grad_norms]),
                hovertemplate=(
                    "Iteración %{x}<br>"
                    "f(x)=%{y:.8e}<br>"
                    "‖∇f‖=%{customdata[0]:.6e}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            xaxis_title="Iteración k",
            yaxis_title="f(xₖ)",
            height=430,
            hovermode="closest",
            margin=dict(l=10, r=10, t=35, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)


def plot_alpha_wolfe(records: List[IterRecord]):
    step_records = [r for r in records if r.alpha is not None]

    if not step_records:
        return

    st.subheader("Tamaño de paso α y condiciones de Wolfe")

    steps = [r.iteracion for r in step_records]
    alphas = [float(r.alpha) for r in step_records]
    w1_vals = [bool(r.wolfe_armijo) for r in step_records]
    w2_vals = [bool(r.wolfe_curvatura) for r in step_records]
    f_vals = [r.f for r in step_records]
    grad_norms = [r.grad_norm for r in step_records]

    alpha_iniciales = [r.alpha_inicial for r in step_records]
    internas = [r.total_internas for r in step_records]
    wolfe_iters = [r.wolfe_iters for r in step_records]
    zoom_iters = [r.zoom_iters for r in step_records]
    modos_alpha = [r.modo_alpha for r in step_records]

    n_steps = len(step_records)
    n_w1 = sum(w1_vals)
    n_w2 = sum(w2_vals)
    n_both = sum(a and b for a, b in zip(w1_vals, w2_vals))

    c1, c2, c3 = st.columns(3)
    c1.metric("Armijo", f"{n_w1}/{n_steps}")
    c2.metric("Curvatura", f"{n_w2}/{n_steps}")
    c3.metric("Ambas", f"{n_both}/{n_steps}")

    colors = ["green" if (a and b) else "orange" if a else "red" for a, b in zip(w1_vals, w2_vals)]

    positive_alphas = [a for a in alphas if a > 0]
    alpha_min_pos = min(positive_alphas, default=1.0)
    alpha_max = max(positive_alphas, default=1.0)
    alpha_range = alpha_max / (alpha_min_pos + 1e-15)
    use_log_alpha = all(a > 0 for a in alphas) and (alpha_range > 100 or alpha_max < 1e-2)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=alphas,
            mode="lines+markers",
            name="α aceptado",
            marker=dict(color=colors, size=10, line=dict(width=0.5)),
            customdata=np.array(
                [
                    [
                        "Sí" if w1 else "No",
                        "Sí" if w2 else "No",
                        fval,
                        gval,
                        alpha_ini,
                        total_int,
                        w_iter,
                        z_iter,
                        modo,
                    ]
                    for w1, w2, fval, gval, alpha_ini, total_int, w_iter, z_iter, modo
                    in zip(
                        w1_vals,
                        w2_vals,
                        f_vals,
                        grad_norms,
                        alpha_iniciales,
                        internas,
                        wolfe_iters,
                        zoom_iters,
                        modos_alpha,
                    )
                ],
                dtype=object,
            ),
            hovertemplate=(
                "Iteración externa %{x}<br>"
                "Modo=%{customdata[8]}<br>"
                "α inicial=%{customdata[4]:.8e}<br>"
                "α aceptado=%{y:.8e}<br>"
                "Iteraciones internas=%{customdata[5]}<br>"
                "Wolfe=%{customdata[6]} · Zoom=%{customdata[7]}<br>"
                "Armijo=%{customdata[0]}<br>"
                "Curvatura=%{customdata[1]}<br>"
                "f después=%{customdata[2]:.8e}<br>"
                "‖∇f‖ después=%{customdata[3]:.6e}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis_title="Iteración externa k",
        yaxis_title="α aceptado",
        yaxis_type="log" if use_log_alpha else "linear",
        title="Paso α — verde: Wolfe OK · naranja: solo Armijo · rojo: revisar",
        height=400,
        hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detalle interno de α por iteración", expanded=False):
        detalle_alpha = {
            "Iteración externa": steps,
            "Modo α": modos_alpha,
            "α inicial": [fmt(a, ".8e") if a is not None else "—" for a in alpha_iniciales],
            "α aceptado": [fmt(a, ".8e") for a in alphas],
            "Iteraciones Wolfe": wolfe_iters,
            "Iteraciones Zoom": zoom_iters,
            "Total internas": internas,
            "Armijo": ["✅" if w else "❌" for w in w1_vals],
            "Curvatura": ["✅" if w else "❌" for w in w2_vals],
        }

        st.dataframe(detalle_alpha, use_container_width=True, hide_index=True)


def eval_grid(obj: Objective, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    try:
        Z = obj.f_raw(X, Y)
        Z = np.asarray(Z, dtype=float)

        if Z.shape != X.shape:
            Z = np.full_like(X, float(Z), dtype=float)

        Z[~np.isfinite(Z)] = np.nan
        return Z

    except Exception:
        Z = np.full_like(X, np.nan, dtype=float)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = obj.f(np.array([X[i, j], Y[i, j]], dtype=float))
                except Exception:
                    pass

        return Z


def add_arrows(fig: go.Figure, tray: np.ndarray, max_arrows: int = 25):
    if len(tray) < 2:
        return

    step = max(1, int(np.ceil((len(tray) - 1) / max_arrows)))
    annotations = []

    for i in range(0, len(tray) - 1, step):
        x0, y0 = tray[i, 0], tray[i, 1]
        x1, y1 = tray[i + 1, 0], tray[i + 1, 1]

        if np.all(np.isfinite([x0, y0, x1, y1])):
            annotations.append(
                dict(
                    x=x1,
                    y=y1,
                    ax=x0,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.1,
                    arrowwidth=1.2,
                    opacity=0.75,
                )
            )

    fig.update_layout(annotations=annotations)


def plot_geometry(
    obj: Objective,
    records: List[IterRecord],
    contour_mode: str,
    contour_style: str,
    show_surface: bool,
):
    tray = np.array([r.x for r in records], dtype=float)
    f_vals = np.array([r.f for r in records], dtype=float)
    grad_vals = np.array([r.grad_norm for r in records], dtype=float)
    n = tray.shape[1]

    if n == 1:
        st.subheader("Función y trayectoria 1D")
        xs_path = tray[:, 0]
        lo, hi = robust_range(xs_path, min_pad=1.0)
        xs = np.linspace(lo, hi, 500)
        ys = np.full_like(xs, np.nan, dtype=float)

        for i, xv in enumerate(xs):
            try:
                ys[i] = obj.f(np.array([xv], dtype=float))
            except Exception:
                pass

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="f(x1)",
                hovertemplate="x1=%{x:.8g}<br>f=%{y:.8e}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xs_path,
                y=f_vals,
                mode="lines+markers",
                name="Trayectoria",
                customdata=np.column_stack([np.arange(len(records)), grad_vals]),
                hovertemplate=(
                    "Iteración %{customdata[0]}<br>"
                    "x1=%{x:.8g}<br>"
                    "f=%{y:.8e}<br>"
                    "‖∇f‖=%{customdata[1]:.6e}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[xs_path[0]],
                y=[f_vals[0]],
                mode="markers",
                marker=dict(size=14, symbol="star"),
                name="Inicio",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[xs_path[-1]],
                y=[f_vals[-1]],
                mode="markers",
                marker=dict(size=14, symbol="diamond"),
                name="Punto final",
            )
        )

        fig.update_layout(
            xaxis_title="x1",
            yaxis_title="f(x1)",
            height=520,
            hovermode="closest",
            margin=dict(l=10, r=10, t=35, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)
        return

    if n == 2:
        st.subheader("Trayectoria sobre curvas de nivel")

        xlo, xhi = robust_range(tray[:, 0], min_pad=1.0)
        ylo, yhi = robust_range(tray[:, 1], min_pad=1.0)
        xx = np.linspace(xlo, xhi, min(MAX_GRID_POINTS, 120))
        yy = np.linspace(ylo, yhi, min(MAX_GRID_POINTS, 120))
        X, Y = np.meshgrid(xx, yy)
        Z = eval_grid(obj, X, Y)
        z_finite = Z[np.isfinite(Z)]

        if z_finite.size == 0:
            st.warning("No se pudo calcular el mapa de curvas de nivel en la región mostrada.")
            return

        if contour_mode == "Recorte robusto 2–98%" and z_finite.size > 10:
            z_low, z_high = np.percentile(z_finite, [2, 98])

            if abs(z_high - z_low) < 1e-14:
                z_low, z_high = np.min(z_finite), np.max(z_finite)

            Z_show = np.clip(Z, z_low, z_high)
            subtitle = "Visualización robusta para resaltar la geometría local."
            z_label = "valor visualizado"
        else:
            Z_show = Z
            subtitle = "Visualización con los valores reales calculados en la malla."
            z_label = "f"

        coloring = "heatmap" if contour_style == "Mapa de calor + líneas" else "lines"

        fig = go.Figure()

        fig.add_trace(
            go.Contour(
                x=xx,
                y=yy,
                z=Z_show,
                colorscale="Viridis",
                contours=dict(showlabels=True, coloring=coloring),
                colorbar=dict(title=z_label),
                name="f(x1,x2)",
                hovertemplate=(
                    "x1=%{x:.6g}<br>"
                    "x2=%{y:.6g}<br>"
                    + z_label
                    + "≈%{z:.6e}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=tray[:, 0],
                y=tray[:, 1],
                mode="lines+markers",
                name="Trayectoria",
                marker=dict(size=7),
                customdata=np.column_stack([np.arange(len(records)), f_vals, grad_vals]),
                hovertemplate=(
                    "Iteración %{customdata[0]}<br>"
                    "x1=%{x:.8g}<br>"
                    "x2=%{y:.8g}<br>"
                    "f=%{customdata[1]:.8e}<br>"
                    "‖∇f‖=%{customdata[2]:.6e}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[tray[0, 0]],
                y=[tray[0, 1]],
                mode="markers",
                marker=dict(size=16, symbol="star"),
                name="Inicio",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[tray[-1, 0]],
                y=[tray[-1, 1]],
                mode="markers",
                marker=dict(size=16, symbol="diamond"),
                name="Punto final",
            )
        )

        add_arrows(fig, tray)

        fig.update_layout(
            xaxis_title="x1",
            yaxis_title="x2",
            height=650,
            hovermode="closest",
            margin=dict(l=10, r=10, t=40, b=10),
            title=subtitle,
        )

        st.plotly_chart(fig, use_container_width=True)

        if show_surface:
            st.subheader("Superficie 3D")

            fig3 = go.Figure(
                data=[
                    go.Surface(
                        x=xx,
                        y=yy,
                        z=Z,
                        colorscale="Viridis",
                        showscale=True,
                        opacity=0.88,
                    )
                ]
            )

            fig3.add_trace(
                go.Scatter3d(
                    x=tray[:, 0],
                    y=tray[:, 1],
                    z=f_vals,
                    mode="lines+markers",
                    name="Trayectoria",
                    marker=dict(size=4),
                    line=dict(width=5),
                )
            )

            fig3.update_layout(
                height=650,
                scene=dict(
                    xaxis_title="x1",
                    yaxis_title="x2",
                    zaxis_title="f(x1,x2)",
                ),
                margin=dict(l=10, r=10, t=35, b=10),
            )

            st.plotly_chart(fig3, use_container_width=True)

        return

    st.subheader("Evolución de variables")

    fig = go.Figure()
    iters = [r.iteracion for r in records]
    use_markers = len(records) <= 100

    for k in range(n):
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=tray[:, k],
                mode="lines+markers" if use_markers else "lines",
                name=f"x{k + 1}",
                hovertemplate=f"Iteración %{{x}}<br>x{k + 1}=%{{y:.8g}}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Iteración k",
        yaxis_title="Valor de variable",
        height=430,
        hovermode="closest",
        margin=dict(l=10, r=10, t=35, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Tablas y exportación
# =============================================================================

def records_to_csv(records: List[IterRecord]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    max_dim = max((len(r.x) for r in records), default=0)

    header = (
        ["iteracion"]
        + [f"x{k + 1}" for k in range(max_dim)]
        + [
            "f(x)",
            "norma_gradiente",
            "alpha_inicial",
            "alpha_aceptado",
            "iteraciones_internas_alpha",
            "iteraciones_wolfe",
            "iteraciones_zoom",
            "modo_alpha",
            "direccion",
            "paso",
            "beta_CG",
            "wolfe_armijo",
            "wolfe_curvatura",
        ]
    )

    writer.writerow(header)

    for r in records:
        row = [r.iteracion]
        row.extend([float(v) for v in r.x])
        row.extend(["" for _ in range(max_dim - len(r.x))])
        row.extend(
            [
                r.f,
                r.grad_norm,
                "" if r.alpha_inicial is None else r.alpha_inicial,
                "" if r.alpha is None else r.alpha,
                r.total_internas,
                r.wolfe_iters,
                r.zoom_iters,
                r.modo_alpha,
                r.direccion,
                r.metodo_paso,
                "" if r.beta is None else r.beta,
                "" if r.wolfe_armijo is None else bool(r.wolfe_armijo),
                "" if r.wolfe_curvatura is None else bool(r.wolfe_curvatura),
            ]
        )

        writer.writerow(row)

    return buffer.getvalue()


def show_history(records: List[IterRecord]):
    with st.expander("Ver historial completo de iteraciones", expanded=False):
        table = {
            "Iteración": [r.iteracion for r in records],
            "x": [vector_to_string(r.x, 8) for r in records],
            "f(x)": [fmt(r.f, ".8e") for r in records],
            "‖∇f‖": [fmt(r.grad_norm, ".6e") for r in records],
            "α inicial": [fmt(r.alpha_inicial, ".8e") if r.alpha_inicial is not None else "—" for r in records],
            "α aceptado": [fmt(r.alpha, ".8e") if r.alpha is not None else "—" for r in records],
            "Iter. internas α": [r.total_internas for r in records],
            "Wolfe": [r.wolfe_iters for r in records],
            "Zoom": [r.zoom_iters for r in records],
            "Modo α": [r.modo_alpha for r in records],
            "Dirección": [r.direccion for r in records],
            "Paso": [r.metodo_paso for r in records],
            "β CG": [fmt(r.beta, ".6e") if r.beta is not None else "—" for r in records],
            "Armijo": [
                "✅" if r.wolfe_armijo else "❌" if r.wolfe_armijo is not None else "—"
                for r in records
            ],
            "Curvatura": [
                "✅" if r.wolfe_curvatura else "❌" if r.wolfe_curvatura is not None else "—"
                for r in records
            ],
        }

        st.dataframe(table, use_container_width=True, hide_index=True)

        st.download_button(
            "Descargar historial CSV",
            data=records_to_csv(records).encode("utf-8"),
            file_name="historial_optimizacion.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =============================================================================
# Interfaz
# =============================================================================

def main():
    st.set_page_config(
        page_title="Optimizador de Funciones",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    ensure_default_state()

    st.title("Optimizador de Funciones")
    st.caption("Por métodos de Gradiente, Gradiente Conjugado y Newton")

    with st.expander("Ejemplos", expanded=False):
        ex_col1, ex_col2 = st.columns([3, 1])

        with ex_col1:
            example_name = st.selectbox(
                "Elegir ejemplo",
                list(EXAMPLES.keys()),
                index=list(EXAMPLES.keys()).index(st.session_state.get("example_selected", DEFAULT_EXAMPLE)),
                key="example_selector",
                label_visibility="collapsed",
            )

        with ex_col2:
            if st.button("Cargar", use_container_width=True):
                load_example_into_state(example_name)
                update_c2_recommendation()
                st.rerun()

    st.subheader("Parámetros")

    funcion = st.text_input(
        "Función objetivo",
        key="funcion",
        help="Usa x1, x2, ..., x10. Ejemplos: x1^2+x2^2, sin(x1), exp(x1), x1*x2.",
    )

    col1, col2, col3 = st.columns([1, 1.4, 1.6])

    with col1:
        n_vars = st.number_input("Variables", min_value=1, max_value=10, step=1, key="n_vars")

    with col2:
        metodo = st.selectbox("Método", METHODS, key="metodo")

    with col3:
        punto_inicial = st.text_input(
            "Punto inicial",
            key="punto_inicial",
            help="Valores separados por coma. Ejemplo: 2, -1",
        )

    with st.expander("Ajustes avanzados", expanded=False):
        adv1, adv2, adv3 = st.columns(3)

        with adv1:
            max_iter = st.number_input(
                "Máximo de iteraciones",
                min_value=1,
                max_value=50000,
                step=10,
                key="max_iter",
            )

            tolerancia = st.number_input(
                "Tolerancia",
                min_value=1e-14,
                format="%.2e",
                key="tolerancia",
            )

        with adv2:
            modo_alpha = st.selectbox("Alpha", ALPHA_MODES, key="modo_alpha")

            alpha_fijo = st.number_input(
                "Valor de alpha fijo",
                min_value=1e-12,
                max_value=100.0,
                format="%.6f",
                key="alpha_fijo",
            )

            c1_input = st.number_input(
                "Wolfe c1",
                min_value=1e-8,
                max_value=0.49,
                format="%.4f",
                key="c1_input",
            )

            c2_input = st.number_input(
                "Wolfe c2",
                min_value=0.01,
                max_value=0.99,
                format="%.2f",
                key="c2_input",
            )

            c2_recomendado = 0.4 if st.session_state.get("metodo") == "Gradiente Conjugado" else 0.9

            if abs(float(st.session_state.get("c2_input", c2_recomendado)) - c2_recomendado) > 1e-12:
                if st.button(f"Usar c2 recomendado ({c2_recomendado:.1f})"):
                    st.session_state["c2_input"] = c2_recomendado
                    st.rerun()

        with adv3:
            contour_style = st.selectbox(
                "Estilo del gráfico 2D",
                ["Mapa de calor + líneas", "Solo líneas"],
                index=0,
            )

            show_surface = st.checkbox("Superficie 3D", value=False)

    max_iter = st.session_state.get("max_iter", 100)
    tolerancia = st.session_state.get("tolerancia", 1e-8)
    c1_input = st.session_state.get("c1_input", 1e-4)
    c2_input = st.session_state.get("c2_input", 0.9)
    modo_alpha = st.session_state.get("modo_alpha", "Automático Wolfe")
    alpha_fijo = st.session_state.get("alpha_fijo", 0.1)

    contour_mode = "Recorte robusto 2–98%"
    contour_style = locals().get("contour_style", "Mapa de calor + líneas")
    show_surface = locals().get("show_surface", False)

    run = st.button("Optimizar", type="primary", use_container_width=True)

    if not run:
        st.stop()

    try:
        n = int(n_vars)
        max_it = int(max_iter)
        tol = float(tolerancia)
        c1 = float(c1_input)
        c2 = float(c2_input)
        modo_alpha = str(modo_alpha)
        alpha_fijo = float(alpha_fijo)

        if not (0 < c1 < c2 < 1):
            st.error(f"Los parámetros Wolfe deben cumplir 0 < c1 < c2 < 1. Tienes c1={c1}, c2={c2}.")
            st.stop()

        if tol <= 0:
            st.error("La tolerancia debe ser estrictamente positiva.")
            st.stop()

        if modo_alpha == "Alpha fijo" and alpha_fijo <= 0:
            st.error("El alpha fijo debe ser estrictamente positivo.")
            st.stop()

        obj = parse_objective(funcion, n)

        parts = [v.strip() for v in punto_inicial.split(",") if v.strip()]

        if len(parts) != n:
            st.error(f"El punto de partida debe tener {n} valores. Tiene {len(parts)}.")
            st.stop()

        x0 = np.array([float(v) for v in parts], dtype=float)

        if not np.all(np.isfinite(x0)):
            st.error("El punto de partida contiene valores no finitos.")
            st.stop()

        _ = obj.f(x0)
        _ = obj.grad(x0)

        if metodo == "Newton":
            _ = obj.hess(x0)

        result = optimize_objective(
            obj=obj,
            x0=x0,
            method=metodo,
            max_iter=max_it,
            tol=tol,
            c1=c1,
            c2=c2,
            modo_alpha=modo_alpha,
            alpha_fijo=alpha_fijo,
        )

        records = result.records
        final = records[-1]
        best = records[result.best_index]

        st.divider()

        if result.converged and result.hessian_class == "minimo_local":
            st.success(f"Resultado encontrado en {final.iteracion} iteraciones.")

        elif result.converged:
            st.warning(f"Se alcanzó la tolerancia del gradiente. {result.hessian_message}")

        else:
            st.warning(f"Proceso detenido en la iteración {final.iteracion}. Motivo: {result.stop_reason}.")

            if metodo == "Gradiente":
                st.info("Gradiente puede ser lento en funciones mal condicionadas. Prueba Gradiente Conjugado o Newton.")

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Iteraciones", final.iteracion)
        cB.metric("f(x)", fmt(final.f))
        cC.metric("Error final ‖∇f‖", fmt(final.grad_norm, ".2e"))
        cD.metric("Tolerancia", f"{tol:.1e}")

        st.write(f"**Punto mínimo/candidato encontrado:** `{vector_to_string(final.x, 10)}`")

        if modo_alpha == "Alpha fijo":
            st.write(f"**Alpha utilizado:** fijo, `α = {alpha_fijo}`")
        else:
            st.write("**Alpha utilizado:** automático por búsqueda de línea Wolfe")

        if result.best_index != len(records) - 1:
            st.write(f"**Mejor punto observado:** `{vector_to_string(best.x, 10)}` con `f={fmt(best.f)}`")

        tab_res, tab_graph, tab_extra = st.tabs(["Resumen", "Gráficos", "Detalles"])

        with tab_res:
            st.write("**Clasificación:**", result.hessian_message)

            if result.eigvals is not None:
                st.write(f"Valores propios de la Hessiana: `{vector_to_string(result.eigvals, 8)}`")

            plot_convergence(records, tol)

        with tab_graph:
            plot_geometry(obj, records, contour_mode, contour_style, show_surface)

        with tab_extra:
            with st.expander("Función y derivadas", expanded=False):
                st.write("**Función objetivo**")
                st.latex(sp.latex(obj.f_sym))

                st.write("**Gradiente**")
                st.latex(sp.latex(sp.Matrix(obj.grad_sym)))

                if n <= 3:
                    st.write("**Hessiana**")
                    st.latex(sp.latex(sp.Matrix(obj.hess_sym)))
                else:
                    st.caption("La Hessiana se calculó correctamente; no se muestra completa para conservar claridad visual.")

            with st.expander("Paso α y condiciones de Wolfe", expanded=False):
                plot_alpha_wolfe(records)

            show_history(records)

    except ValueError as e:
        st.error(str(e))

    except Exception as e:
        import traceback

        st.error(f"Error inesperado: {e}")

        with st.expander("Detalle técnico"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
