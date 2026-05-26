import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Optimizador de Funciones", layout="wide")

SP_TRANSFORMS = standard_transformations + (convert_xor,)

LOG_FLOOR = 1e-16
ALPHA_EPS = 1e-15
MAX_LINESEARCH_ITERS = 60
MAX_ZOOM_ITERS = 60
MAX_ALPHA = 10.0
MAX_GRID_POINTS = 140


@dataclass
class IterRecord:
    iteracion: int
    x: np.ndarray
    f: float
    grad_norm: float
    alpha: Optional[float]
    direccion: Optional[str]
    wolfe_armijo: Optional[bool]
    wolfe_curvatura: Optional[bool]
    metodo_paso: Optional[str]


# -----------------------------------------------------------------------------
# Utilidades numéricas y de formato
# -----------------------------------------------------------------------------


def fmt(v, fmt_spec: str = ".4e") -> str:
    """Formato seguro para valores escalares que pueden ser NaN/Inf."""
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
    """Convierte un valor a float y rechaza NaN/Inf o valores complejos."""
    val = np.asarray(value)
    if np.iscomplexobj(val):
        if np.max(np.abs(np.imag(val))) > 1e-12:
            raise ValueError("la evaluación produjo un valor complejo")
        val = np.real(val)
    out = float(np.asarray(val, dtype=float))
    if not np.isfinite(out):
        raise ValueError("la evaluación produjo NaN o infinito")
    return out


def safe_eval_f(f_raw: Callable, x: np.ndarray) -> float:
    return finite_float(f_raw(*x))


def safe_eval_grad(grad_raw: Sequence[Callable], x: np.ndarray) -> np.ndarray:
    vals = np.array([finite_float(g(*x)) for g in grad_raw], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("el gradiente contiene NaN o infinito")
    return vals


def safe_eval_hess(hess_raw: Sequence[Sequence[Callable]], x: np.ndarray) -> np.ndarray:
    H = np.array([[finite_float(h(*x)) for h in row] for row in hess_raw], dtype=float)
    if not np.all(np.isfinite(H)):
        raise ValueError("la Hessiana contiene NaN o infinito")
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


def robust_range(values: np.ndarray, min_pad: float = 1.0) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -min_pad, min_pad
    if vals.size >= 5:
        lo, hi = np.percentile(vals, [2, 98])
        full_lo, full_hi = np.min(vals), np.max(vals)
        # No ignores entirely the actual endpoints; just dampen extreme jumps.
        lo = min(lo, full_lo)
        hi = max(hi, full_hi)
    else:
        lo, hi = np.min(vals), np.max(vals)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -min_pad, min_pad
    if abs(hi - lo) < 1e-12:
        center = 0.5 * (hi + lo)
        return center - min_pad, center + min_pad
    pad = max(min_pad, 0.25 * abs(hi - lo))
    return lo - pad, hi + pad


def classify_hessian(H: Optional[np.ndarray], eig_tol: float = 1e-8) -> Tuple[str, str, Optional[np.ndarray]]:
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
        return "maximo_local", "Hessiana definida negativa: candidato a máximo local, no a mínimo.", eigvals
    if min_eig < -eig_tol < max_eig:
        return "silla", "Hessiana indefinida: el punto final parece ser un punto silla.", eigvals
    return "degenerado", "Hessiana semidefinida o casi singular: clasificación no concluyente.", eigvals


# -----------------------------------------------------------------------------
# Búsqueda de línea Wolfe robusta
# -----------------------------------------------------------------------------


def make_wolfe_search(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    c1: float,
    c2: float,
):
    def phi(x: np.ndarray, p: np.ndarray, alpha: float) -> float:
        return f(x + alpha * p)

    def dphi(x: np.ndarray, p: np.ndarray, alpha: float) -> float:
        return float(np.dot(grad(x + alpha * p), p))

    def valid_phi(x: np.ndarray, p: np.ndarray, alpha: float) -> Tuple[bool, float]:
        try:
            val = phi(x, p, alpha)
            return np.isfinite(val), val
        except Exception:
            return False, float("nan")

    def zoom(alpha_lo: float, alpha_hi: float, x: np.ndarray, p: np.ndarray, phi0: float, dphi0: float) -> float:
        phi_lo_ok, phi_lo = valid_phi(x, p, alpha_lo)
        if not phi_lo_ok:
            return 0.0

        for _ in range(MAX_ZOOM_ITERS):
            if abs(alpha_hi - alpha_lo) < 1e-12:
                break
            alpha = 0.5 * (alpha_lo + alpha_hi)
            ok_a, phi_a = valid_phi(x, p, alpha)
            if not ok_a:
                alpha_hi = alpha
                continue

            if phi_a > phi0 + c1 * alpha * dphi0 or phi_a >= phi_lo:
                alpha_hi = alpha
            else:
                try:
                    dphi_a = dphi(x, p, alpha)
                except Exception:
                    alpha_hi = alpha
                    continue
                if abs(dphi_a) <= -c2 * dphi0:
                    return alpha
                if dphi_a * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha
                phi_lo = phi_a
        return max(0.0, 0.5 * (alpha_lo + alpha_hi))

    def wolfe_search(x: np.ndarray, p: np.ndarray) -> float:
        try:
            phi0 = f(x)
            g0 = grad(x)
            dphi0 = float(np.dot(g0, p))
        except Exception:
            return 0.0

        if not np.isfinite(phi0) or not np.isfinite(dphi0) or dphi0 >= 0:
            return 0.0

        alpha_prev = 0.0
        phi_prev = phi0
        alpha = 1.0

        for i in range(MAX_LINESEARCH_ITERS):
            ok_a, phi_a = valid_phi(x, p, alpha)
            if not ok_a:
                # Fuera de dominio o desborde: reduce el paso hasta volver a una zona válida.
                alpha_hi = alpha
                alpha_lo = alpha_prev
                return zoom(alpha_lo, alpha_hi, x, p, phi0, dphi0)

            if phi_a > phi0 + c1 * alpha * dphi0 or (i > 0 and phi_a >= phi_prev):
                return zoom(alpha_prev, alpha, x, p, phi0, dphi0)

            try:
                dphi_a = dphi(x, p, alpha)
            except Exception:
                return zoom(alpha_prev, alpha, x, p, phi0, dphi0)

            if not np.isfinite(dphi_a):
                return zoom(alpha_prev, alpha, x, p, phi0, dphi0)

            if abs(dphi_a) <= -c2 * dphi0:
                return alpha
            if dphi_a >= 0:
                return zoom(alpha, alpha_prev, x, p, phi0, dphi0)

            alpha_prev = alpha
            phi_prev = phi_a
            alpha = min(2.0 * alpha, MAX_ALPHA)

        return alpha

    def verificar_wolfe(x: np.ndarray, p: np.ndarray, alpha: float) -> Tuple[bool, bool]:
        try:
            phi0 = f(x)
            dphi0 = float(np.dot(grad(x), p))
            phi_a = f(x + alpha * p)
            dphi_a = float(np.dot(grad(x + alpha * p), p))
            if not all(np.isfinite(v) for v in [phi0, dphi0, phi_a, dphi_a]):
                return False, False
            w1 = bool(phi_a <= phi0 + c1 * alpha * dphi0)
            w2 = bool(abs(dphi_a) <= c2 * abs(dphi0))
            return w1, w2
        except Exception:
            return False, False

    return wolfe_search, verificar_wolfe


# -----------------------------------------------------------------------------
# Gráficos
# -----------------------------------------------------------------------------


def plot_convergence(records: List[IterRecord], tol: float):
    st.subheader("Convergencia")

    iters = [r.iteracion for r in records]
    grad_norms = [r.grad_norm for r in records]
    f_vals = [r.f for r in records]
    use_markers = len(records) <= 80

    tab_grad_log, tab_grad_lin, tab_fx = st.tabs([
        "📉 ‖∇f‖ log",
        "📈 ‖∇f‖ lineal",
        "🎯 Valor objetivo f(x)",
    ])

    def make_grad_fig(y_data, log_scale: bool):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iters,
            y=y_data,
            mode="lines+markers" if use_markers else "lines",
            name="‖∇f‖",
            customdata=np.column_stack([f_vals]),
            hovertemplate=(
                "Iteración %{x}<br>"
                "‖∇f‖=%{y:.6e}<br>"
                "f(x)=%{customdata[0]:.6e}<extra></extra>"
            ),
        ))
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
        st.plotly_chart(make_grad_fig(safe_log_values(grad_norms), True), use_container_width=True)
        st.caption("Los valores 0 se muestran como 1e-16 para permitir la escala logarítmica.")

    with tab_grad_lin:
        st.plotly_chart(make_grad_fig(grad_norms, False), use_container_width=True)

    with tab_fx:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iters,
            y=f_vals,
            mode="lines+markers" if use_markers else "lines",
            name="f(xₖ)",
            customdata=np.column_stack([grad_norms]),
            hovertemplate=(
                "Iteración %{x}<br>"
                "f(x)=%{y:.8e}<br>"
                "‖∇f‖=%{customdata[0]:.6e}<extra></extra>"
            ),
        ))
        fig.update_layout(
            xaxis_title="Iteración k",
            yaxis_title="f(xₖ)",
            height=430,
            hovermode="closest",
            margin=dict(l=10, r=10, t=35, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Este gráfico permite verificar si la función objetivo disminuye durante el proceso.")


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

    n_steps = len(step_records)
    n_w1 = sum(w1_vals)
    n_w2 = sum(w2_vals)
    n_both = sum(a and b for a, b in zip(w1_vals, w2_vals))

    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("Armijo (c1) ✅", f"{n_w1}/{n_steps} pasos")
    wc2.metric("Curvatura (c2) ✅", f"{n_w2}/{n_steps} pasos")
    wc3.metric("Ambas condiciones", f"{n_both}/{n_steps} pasos")

    colors = ["green" if (a and b) else "orange" if a else "red" for a, b in zip(w1_vals, w2_vals)]
    alpha_min_pos = min((a for a in alphas if a > 0), default=1.0)
    alpha_range = max(alphas) / (alpha_min_pos + 1e-15)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps,
        y=alphas,
        mode="lines+markers",
        name="α",
        marker=dict(color=colors, size=10, line=dict(width=0.5)),
        customdata=np.column_stack([w1_vals, w2_vals, f_vals, grad_norms]),
        hovertemplate=(
            "Paso %{x}<br>"
            "α=%{y:.8e}<br>"
            "Armijo=%{customdata[0]}<br>"
            "Curvatura=%{customdata[1]}<br>"
            "f después=%{customdata[2]:.8e}<br>"
            "‖∇f‖ después=%{customdata[3]:.6e}<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title="Paso k",
        yaxis_title="αₖ",
        yaxis_type="log" if alpha_range > 100 else "linear",
        title="Paso α — verde: Wolfe OK · naranja: solo Armijo · rojo: fallo",
        height=400,
        hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    if n_steps <= 100:
        with st.expander("Ver detalle por iteración"):
            table = {
                "Paso": steps,
                "f(xₖ) después": [fmt(v, ".6e") for v in f_vals],
                "‖∇f(xₖ)‖ después": [fmt(v, ".4e") for v in grad_norms],
                "α": [fmt(a, ".8f") for a in alphas],
                "Armijo ✓": ["✅" if w else "❌" for w in w1_vals],
                "Curvatura ✓": ["✅" if w else "❌" for w in w2_vals],
                "Dirección": [r.direccion or "—" for r in step_records],
            }
            st.dataframe(table, use_container_width=True, hide_index=True)


def eval_grid_2d(f_lam: Callable, f: Callable[[np.ndarray], float], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    try:
        Z = f_lam(X, Y)
        Z = np.asarray(Z, dtype=float)
        if Z.shape != X.shape:
            Z = np.full_like(X, float(Z), dtype=float)
        return Z
    except Exception:
        Z = np.full_like(X, np.nan, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = f(np.array([X[i, j], Y[i, j]], dtype=float))
                except Exception:
                    pass
        return Z


def add_direction_arrows(fig: go.Figure, tray: np.ndarray, max_arrows: int = 25):
    if len(tray) < 2:
        return
    step = max(1, (len(tray) - 1) // max_arrows)
    for i in range(0, len(tray) - 1, step):
        x0, y0 = tray[i]
        x1, y1 = tray[i + 1]
        if np.all(np.isfinite([x0, y0, x1, y1])):
            fig.add_annotation(
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
                arrowsize=1,
                arrowwidth=1,
                opacity=0.65,
            )


def plot_geometry(
    n: int,
    records: List[IterRecord],
    f_lam: Callable,
    f: Callable[[np.ndarray], float],
):
    tray = np.array([r.x for r in records], dtype=float)
    f_vals = np.array([r.f for r in records], dtype=float)
    grad_norms = np.array([r.grad_norm for r in records], dtype=float)
    iters = np.array([r.iteracion for r in records], dtype=int)

    if n == 1:
        st.subheader("Función y trayectoria (1D)")
        tray_1d = tray[:, 0]
        lo, hi = robust_range(tray_1d, min_pad=1.0)
        xs = np.linspace(lo, hi, 500)

        ys = np.full_like(xs, np.nan, dtype=float)
        try:
            ys_try = f_lam(xs)
            ys_arr = np.asarray(ys_try, dtype=float)
            ys = ys_arr if ys_arr.shape == xs.shape else np.full_like(xs, float(ys_try))
        except Exception:
            for i, xv in enumerate(xs):
                try:
                    ys[i] = f(np.array([xv], dtype=float))
                except Exception:
                    pass

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x₁)"))
        fig.add_trace(go.Scatter(
            x=tray_1d,
            y=f_vals,
            mode="lines+markers",
            name="Trayectoria",
            customdata=np.column_stack([iters, grad_norms]),
            hovertemplate=(
                "Iteración %{customdata[0]}<br>"
                "x₁=%{x:.8g}<br>"
                "f(x)=%{y:.8e}<br>"
                "‖∇f‖=%{customdata[1]:.6e}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(x=[tray_1d[0]], y=[f_vals[0]], mode="markers", marker=dict(size=15, symbol="star"), name="Inicio"))
        fig.add_trace(go.Scatter(x=[tray_1d[-1]], y=[f_vals[-1]], mode="markers", marker=dict(size=15, symbol="star"), name="Punto final"))
        fig.update_layout(xaxis_title="x₁", yaxis_title="f(x₁)", height=520, hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)

    elif n == 2:
        st.subheader("Trayectoria sobre curvas de nivel")
        col1, col2, col3 = st.columns([1.1, 1.1, 1.2])
        with col1:
            contour_mode = st.selectbox(
                "Visualización del contorno",
                ["Recortado 2%-98%", "Valores reales"],
                index=0,
                help="El recorte mejora la lectura cuando hay valores extremos.",
            )
        with col2:
            contour_style = st.selectbox("Estilo", ["Mapa de calor + líneas", "Solo líneas"], index=0)
        with col3:
            show_surface = st.checkbox("Mostrar superficie 3D", value=False)

        x_lo, x_hi = robust_range(tray[:, 0], min_pad=1.0)
        y_lo, y_hi = robust_range(tray[:, 1], min_pad=1.0)
        grid_n = 100 if len(records) <= 200 else 80
        grid_n = min(grid_n, MAX_GRID_POINTS)
        xx = np.linspace(x_lo, x_hi, grid_n)
        yy = np.linspace(y_lo, y_hi, grid_n)
        X, Y = np.meshgrid(xx, yy)
        Z = eval_grid_2d(f_lam, f, X, Y)

        z_finite = Z[np.isfinite(Z)]
        if z_finite.size == 0:
            st.warning("No se pudo calcular el mapa de curvas de nivel en la región mostrada.")
            return

        Z_plot = Z.copy()
        if contour_mode == "Recortado 2%-98%":
            z_low, z_high = np.percentile(z_finite, [2, 98])
            if abs(z_high - z_low) < 1e-12:
                z_low -= 1.0
                z_high += 1.0
            Z_plot = np.clip(Z_plot, z_low, z_high)
            st.caption("El mapa de contorno está recortado entre percentiles 2 y 98 para mejorar la visualización. Los cálculos usan la función real.")

        custom_tray = np.column_stack([iters, f_vals, grad_norms])
        coloring = "heatmap" if contour_style == "Mapa de calor + líneas" else "lines"

        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=xx,
            y=yy,
            z=Z_plot,
            contours=dict(showlabels=True, coloring=coloring),
            colorbar=dict(title="f(x₁,x₂)"),
            name="f(x₁,x₂)",
            hovertemplate="x₁=%{x:.6g}<br>x₂=%{y:.6g}<br>f≈%{z:.6e}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=tray[:, 0],
            y=tray[:, 1],
            mode="lines+markers",
            name="Trayectoria",
            marker=dict(size=7),
            customdata=custom_tray,
            hovertemplate=(
                "Iteración %{customdata[0]}<br>"
                "x₁=%{x:.8g}<br>"
                "x₂=%{y:.8g}<br>"
                "f(x)=%{customdata[1]:.8e}<br>"
                "‖∇f‖=%{customdata[2]:.6e}<extra></extra>"
            ),
        ))
        add_direction_arrows(fig, tray)
        fig.add_trace(go.Scatter(x=[tray[0, 0]], y=[tray[0, 1]], mode="markers", marker=dict(size=16, symbol="star"), name="Inicio"))
        fig.add_trace(go.Scatter(x=[tray[-1, 0]], y=[tray[-1, 1]], mode="markers", marker=dict(size=16, symbol="star"), name="Punto final"))
        fig.update_layout(
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=650,
            hovermode="closest",
            margin=dict(l=10, r=10, t=35, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_surface:
            fig3d = go.Figure()
            fig3d.add_trace(go.Surface(x=xx, y=yy, z=Z_plot, name="Superficie", opacity=0.88))
            fig3d.add_trace(go.Scatter3d(
                x=tray[:, 0],
                y=tray[:, 1],
                z=f_vals,
                mode="lines+markers",
                name="Trayectoria",
                marker=dict(size=4),
                line=dict(width=5),
                customdata=custom_tray,
                hovertemplate=(
                    "Iteración %{customdata[0]}<br>"
                    "x₁=%{x:.8g}<br>"
                    "x₂=%{y:.8g}<br>"
                    "f(x)=%{z:.8e}<br>"
                    "‖∇f‖=%{customdata[2]:.6e}<extra></extra>"
                ),
            ))
            fig3d.update_layout(
                scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f(x₁,x₂)"),
                height=650,
                margin=dict(l=0, r=0, t=35, b=0),
            )
            st.plotly_chart(fig3d, use_container_width=True)

    else:
        st.subheader("Evolución de variables durante la optimización")
        fig = go.Figure()
        for k in range(n):
            fig.add_trace(go.Scatter(
                x=iters,
                y=tray[:, k],
                mode="lines+markers" if len(records) <= 80 else "lines",
                name=f"x{k + 1}",
                customdata=np.column_stack([f_vals, grad_norms]),
                hovertemplate=(
                    "Iteración %{x}<br>"
                    f"x{k + 1}=%{{y:.8g}}<br>"
                    "f(x)=%{customdata[0]:.8e}<br>"
                    "‖∇f‖=%{customdata[1]:.6e}<extra></extra>"
                ),
            ))
        fig.update_layout(xaxis_title="Iteración k", yaxis_title="Valor de variable", height=430)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Para n > 2 no hay mapa de curvas de nivel directo; se muestra la evolución de cada variable.")


# -----------------------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------------------

st.title("Optimizador de Funciones")
st.caption("Métodos de optimización sin restricciones con búsqueda de línea Wolfe.")

with st.expander("ℹ️ Alcance de la herramienta", expanded=True):
    st.markdown(
        """
Esta app busca **puntos estacionarios y candidatos a mínimos locales** de funciones suaves sin restricciones.  
No garantiza mínimos globales, no maneja restricciones explícitas y puede fallar o detenerse en funciones no diferenciables, discontinuas, con singularidades o sin mínimo finito.
"""
    )

st.subheader("Parámetros de entrada")

examples: Dict[str, Dict[str, object]] = {
    "Cuadrática simple": {"f": "x1^2 + x2^2", "n": 2, "x0": "2, 2", "metodo": "Newton"},
    "Cuadrática anisotrópica": {"f": "100*x1^2 + x2^2", "n": 2, "x0": "2, 2", "metodo": "Gradiente Conjugado"},
    "Rosenbrock": {"f": "100*(x2 - x1^2)^2 + (1 - x1)^2", "n": 2, "x0": "-1.2, 1", "metodo": "Newton"},
    "Himmelblau": {"f": "(x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2", "n": 2, "x0": "-3, 3", "metodo": "Newton"},
    "1D simple": {"f": "(x1 - 5)^2", "n": 1, "x0": "0", "metodo": "Newton"},
    "5D cuadrática": {"f": "x1^2 + 2*x2^2 + 3*x3^2 + 4*x4^2 + 5*x5^2", "n": 5, "x0": "1, 1, 1, 1, 1", "metodo": "Gradiente Conjugado"},
}

with st.sidebar:
    st.header("Ejemplos")
    example_name = st.selectbox("Cargar ejemplo", list(examples.keys()), index=0)
    ex = examples[example_name]
    st.caption("Puedes modificar cualquier campo después de cargar el ejemplo.")

col1, col2 = st.columns([2, 1])
with col1:
    funcion = st.text_input("Función objetivo", value=str(ex["f"]), help="Usa x1, x2, ... Puedes escribir ^ como potencia.")
with col2:
    n_vars = st.number_input("Número de variables", min_value=1, max_value=5, value=int(ex["n"]), step=1)

metodos = ["Gradiente", "Gradiente Conjugado", "Newton"]
default_metodo = str(ex["metodo"])
metodo = st.selectbox("Método de optimización", metodos, index=metodos.index(default_metodo) if default_metodo in metodos else 0)

col3, col4, col5 = st.columns(3)
with col3:
    punto_inicial = st.text_input("Punto de partida", value=str(ex["x0"]), help="Valores separados por comas, por ejemplo: 2, -1")
with col4:
    max_iter = st.number_input("Máximo de iteraciones", min_value=1, max_value=100000, value=500, step=50)
with col5:
    tolerancia = st.number_input("Tolerancia de convergencia", min_value=1e-14, value=1e-6, format="%.2e")

col6, col7 = st.columns(2)
with col6:
    c1_input = st.number_input("Parámetro Wolfe c1 (Armijo)", min_value=1e-12, max_value=0.999, value=1e-4, format="%.4f")
with col7:
    c2_input = st.number_input("Parámetro Wolfe c2 (curvatura)", min_value=1e-12, max_value=0.999999, value=0.9, format="%.2f")

st.divider()

if st.button("Optimizar", type="primary"):
    try:
        n = int(n_vars)
        c1_val = float(c1_input)
        c2_val = float(c2_input)
        tol = float(tolerancia)
        max_it = int(max_iter)

        if not (0 < c1_val < c2_val < 1):
            st.error(f"Los parámetros Wolfe deben cumplir 0 < c1 < c2 < 1. Tienes c1={c1_val}, c2={c2_val}.")
            st.stop()
        if tol <= 0:
            st.error("La tolerancia debe ser estrictamente positiva.")
            st.stop()

        variables = sp.symbols(f"x1:{n + 1}")
        if n == 1 and not isinstance(variables, tuple):
            variables = (variables,)

        try:
            f_sym = parse_expr(funcion, transformations=SP_TRANSFORMS, evaluate=True)
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")
            st.stop()

        extras = f_sym.free_symbols - set(variables)
        if extras:
            st.error(f"La función usa variables no permitidas: {extras}. Solo puedes usar x1,...,x{n}.")
            st.stop()

        if len(f_sym.free_symbols) == 0:
            st.warning("La función no depende de las variables. Cualquier punto tiene gradiente cero.")

        try:
            grad_sym = [sp.diff(f_sym, v) for v in variables]
            hess_sym = [[sp.diff(g, v) for v in variables] for g in grad_sym]
            f_lam = sp.lambdify(variables, f_sym, "numpy")
            grad_lams = [sp.lambdify(variables, g, "numpy") for g in grad_sym]
            hess_lams = [[sp.lambdify(variables, h, "numpy") for h in row] for row in hess_sym]
        except Exception as e:
            st.error(f"No se pudieron calcular las derivadas simbólicas: {e}")
            st.stop()

        with st.expander("Ver derivadas simbólicas"):
            st.write("**Función:**")
            st.latex(sp.latex(f_sym))
            st.write("**Gradiente:**")
            st.latex(sp.latex(sp.Matrix(grad_sym)))
            if n <= 3:
                st.write("**Hessiana:**")
                st.latex(sp.latex(sp.Matrix(hess_sym)))
            else:
                st.caption("La Hessiana se calculó, pero no se muestra completa para evitar saturar la interfaz.")

        def f(x: np.ndarray) -> float:
            return safe_eval_f(f_lam, np.asarray(x, dtype=float))

        def grad(x: np.ndarray) -> np.ndarray:
            return safe_eval_grad(grad_lams, np.asarray(x, dtype=float))

        def hess(x: np.ndarray) -> np.ndarray:
            return safe_eval_hess(hess_lams, np.asarray(x, dtype=float))

        partes = [v.strip() for v in punto_inicial.split(",") if v.strip() != ""]
        if len(partes) != n:
            st.error(f"El punto de partida debe tener {n} valores. Tiene {len(partes)}.")
            st.stop()
        try:
            x0 = np.array([float(v) for v in partes], dtype=float)
        except ValueError as e:
            st.error(f"El punto de partida tiene valores no numéricos: {e}")
            st.stop()
        if not np.all(np.isfinite(x0)):
            st.error("El punto de partida contiene valores no finitos (NaN/Inf).")
            st.stop()

        try:
            f0 = f(x0)
            g0 = grad(x0)
            if metodo == "Newton":
                _ = hess(x0)
        except Exception as e:
            st.error(f"No se pudo evaluar la función o sus derivadas en x0: {e}")
            st.stop()

        wolfe_search, verificar_wolfe = make_wolfe_search(f, grad, c1_val, c2_val)

        records: List[IterRecord] = []
        x = x0.copy()
        razon_paro = "máximo de iteraciones"

        def add_record(iteracion: int, x_val: np.ndarray, alpha=None, direccion=None, w1=None, w2=None, metodo_paso=None):
            fx = f(x_val)
            gx = grad(x_val)
            records.append(IterRecord(
                iteracion=iteracion,
                x=x_val.copy(),
                f=float(fx),
                grad_norm=float(np.linalg.norm(gx)),
                alpha=alpha,
                direccion=direccion,
                wolfe_armijo=w1,
                wolfe_curvatura=w2,
                metodo_paso=metodo_paso,
            ))

        add_record(0, x)

        if records[-1].grad_norm < tol:
            razon_paro = "el punto inicial ya cumple la tolerancia"
        elif metodo == "Gradiente":
            for k in range(1, max_it + 1):
                try:
                    g = grad(x)
                except Exception:
                    razon_paro = "gradiente no finito"
                    break
                if np.linalg.norm(g) < tol:
                    razon_paro = "tolerancia alcanzada"
                    break
                p = -g
                alpha = wolfe_search(x, p)
                if alpha <= ALPHA_EPS:
                    razon_paro = "α ≈ 0: la búsqueda de línea no encontró un paso válido"
                    break
                w1, w2 = verificar_wolfe(x, p, alpha)
                x_new = x + alpha * p
                try:
                    add_record(k, x_new, alpha, "-∇f", w1, w2, "Gradiente")
                except Exception:
                    razon_paro = "el nuevo punto no es evaluable o salió del dominio"
                    break
                x = x_new
                if records[-1].grad_norm < tol:
                    razon_paro = "tolerancia alcanzada"
                    break

        elif metodo == "Gradiente Conjugado":
            try:
                g = grad(x)
            except Exception:
                st.error("El gradiente inicial no es finito.")
                st.stop()
            p = -g
            for k in range(1, max_it + 1):
                if np.linalg.norm(g) < tol:
                    razon_paro = "tolerancia alcanzada"
                    break
                if np.dot(p, g) >= 0 or not np.all(np.isfinite(p)):
                    p = -g
                alpha = wolfe_search(x, p)
                metodo_paso = "CG"
                if alpha <= ALPHA_EPS:
                    p = -g
                    alpha = wolfe_search(x, p)
                    metodo_paso = "reinicio a -∇f"
                    if alpha <= ALPHA_EPS:
                        razon_paro = "α ≈ 0: la búsqueda de línea no encontró un paso válido"
                        break
                w1, w2 = verificar_wolfe(x, p, alpha)
                x_new = x + alpha * p
                try:
                    g_new = grad(x_new)
                    add_record(k, x_new, alpha, "CG" if metodo_paso == "CG" else "-∇f", w1, w2, metodo_paso)
                except Exception:
                    razon_paro = "el nuevo punto no es evaluable o salió del dominio"
                    break
                denom = float(np.dot(g, g))
                beta = max(0.0, float(np.dot(g_new, g_new - g)) / denom) if denom > 1e-20 else 0.0
                p = -g_new + beta * p
                x, g = x_new, g_new
                if records[-1].grad_norm < tol:
                    razon_paro = "tolerancia alcanzada"
                    break

        elif metodo == "Newton":
            for k in range(1, max_it + 1):
                try:
                    g = grad(x)
                except Exception:
                    razon_paro = "gradiente no finito"
                    break
                if np.linalg.norm(g) < tol:
                    razon_paro = "tolerancia alcanzada"
                    break
                metodo_paso = "Newton"
                direccion = "Newton"
                try:
                    H = hess(x)
                    Hs = 0.5 * (H + H.T)
                    np.linalg.cholesky(Hs)
                    p = -np.linalg.solve(Hs, g)
                    if not np.all(np.isfinite(p)) or np.dot(p, g) >= 0:
                        p = -g
                        metodo_paso = "fallback a -∇f"
                        direccion = "-∇f"
                except Exception:
                    p = -g
                    metodo_paso = "fallback a -∇f"
                    direccion = "-∇f"

                alpha = wolfe_search(x, p)
                if alpha <= ALPHA_EPS and direccion != "-∇f":
                    p = -g
                    alpha = wolfe_search(x, p)
                    metodo_paso = "fallback a -∇f"
                    direccion = "-∇f"
                if alpha <= ALPHA_EPS:
                    razon_paro = "α ≈ 0: la búsqueda de línea no encontró un paso válido"
                    break
                w1, w2 = verificar_wolfe(x, p, alpha)
                x_new = x + alpha * p
                try:
                    add_record(k, x_new, alpha, direccion, w1, w2, metodo_paso)
                except Exception:
                    razon_paro = "el nuevo punto no es evaluable o salió del dominio"
                    break
                x = x_new
                if records[-1].grad_norm < tol:
                    razon_paro = "tolerancia alcanzada"
                    break

        if not records:
            st.error("La optimización no pudo registrar ninguna iteración.")
            st.stop()

        final = records[-1]
        converged = np.isfinite(final.grad_norm) and final.grad_norm < tol

        try:
            H_final = hess(final.x)
        except Exception:
            H_final = None
        clase, mensaje_hess, eigvals = classify_hessian(H_final)

        if converged and clase == "minimo_local":
            st.success(f"✅ Candidato a mínimo local encontrado en {final.iteracion} iteraciones ({razon_paro}).")
        elif converged:
            st.warning(f"⚠️ Se alcanzó ‖∇f‖ < tolerancia, pero la clasificación no confirma un mínimo local. Motivo: {mensaje_hess}")
        else:
            st.warning(f"⚠️ No se alcanzó la tolerancia. Iteraciones: {final.iteracion} · ‖∇f‖ final: {fmt(final.grad_norm)} · motivo: {razon_paro}.")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Iteraciones", final.iteracion)
        col_b.metric("f(x final)", fmt(final.f))
        col_c.metric("‖∇f‖ final", fmt(final.grad_norm, ".2e"))
        col_d.metric("Tolerancia", f"{tol:.1e}")

        st.write(f"**Punto final / candidato:** `{vector_to_string(final.x)}`")
        if eigvals is not None:
            st.write(f"**Clasificación Hessiana:** {mensaje_hess}")
            st.caption("Valores propios Hessiana final: " + vector_to_string(eigvals, precision=6))
        else:
            st.write(f"**Clasificación Hessiana:** {mensaje_hess}")

        plot_convergence(records, tol)
        plot_alpha_wolfe(records)
        plot_geometry(n, records, f_lam, f)

        with st.expander("Tabla completa del historial"):
            table = {
                "Iteración": [r.iteracion for r in records],
                "x": [vector_to_string(r.x, precision=8) for r in records],
                "f(x)": [fmt(r.f, ".8e") for r in records],
                "‖∇f‖": [fmt(r.grad_norm, ".4e") for r in records],
                "α": [fmt(r.alpha, ".8f") if r.alpha is not None else "—" for r in records],
                "Dirección": [r.direccion or "—" for r in records],
                "Armijo": ["✅" if r.wolfe_armijo else "❌" if r.wolfe_armijo is not None else "—" for r in records],
                "Curvatura": ["✅" if r.wolfe_curvatura else "❌" if r.wolfe_curvatura is not None else "—" for r in records],
            }
            st.dataframe(table, use_container_width=True, hide_index=True)

    except Exception as e:
        import traceback
        st.error(f"Error inesperado: {e}")
        with st.expander("Ver detalle técnico"):
            st.code(traceback.format_exc())
