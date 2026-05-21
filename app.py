import streamlit as st
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
)
import plotly.graph_objects as go

# Permite que el usuario escriba x1^2 (en vez de x1**2) y que sympy lo entienda
# como potencia y no como XOR bit a bit
SP_TRANSFORMS = standard_transformations + (convert_xor,)

st.title("Optimizador de Funciones")
st.subheader("Parámetros de entrada")

funcion       = st.text_input("Función objetivo (usa x1, x2, etc.)", value="x1**2 + x2**2")
n_vars        = st.number_input("Número de variables", min_value=1, max_value=5, value=2)
metodo        = st.selectbox("Método de optimización", ["Gradiente", "Gradiente Conjugado", "Newton"])
punto_inicial = st.text_input("Punto de partida (separado por comas)", value="2, 2")
max_iter      = st.number_input("Máximo de iteraciones", min_value=1, max_value=10000, value=200)
tolerancia    = st.number_input("Tolerancia de convergencia", value=1e-6, format="%.2e")
c1_input      = st.number_input("Parámetro Wolfe c1 (Armijo)", value=1e-4, format="%.4f")
c2_input      = st.number_input("Parámetro Wolfe c2 (curvatura)", value=0.9, format="%.2f")

LOG_FLOOR = 1e-16  # piso para representar valores ≤0 en escala log

def fmt(v, fmt_spec=".4e"):
    """Formato seguro para valores que pueden ser NaN/Inf."""
    if v is None:
        return "—"
    try:
        if not np.isfinite(v):
            return "no finito"
        return f"{v:{fmt_spec}}"
    except Exception:
        return str(v)


if st.button("Optimizar"):
    try:
        n       = int(n_vars)
        c1_val  = float(c1_input)
        c2_val  = float(c2_input)
        tol     = float(tolerancia)
        max_it  = int(max_iter)

        # ── Validación de parámetros ───────────────────────────────────────────
        if not (0 < c1_val < c2_val < 1):
            st.error(f"Los parámetros Wolfe deben cumplir 0 < c1 < c2 < 1. "
                     f"Tienes c1={c1_val}, c2={c2_val}.")
            st.stop()

        if tol <= 0:
            st.error("La tolerancia debe ser estrictamente positiva.")
            st.stop()

        # ── Parseo de la función ───────────────────────────────────────────────
        variables = sp.symbols(f'x1:{n+1}')
        if n == 1 and not isinstance(variables, tuple):
            variables = (variables,)

        try:
            f_sym = parse_expr(funcion, transformations=SP_TRANSFORMS)
        except Exception as e:
            st.error(f"No se pudo interpretar la función: {e}")
            st.stop()

        extras = f_sym.free_symbols - set(variables)
        if extras:
            st.error(f"La función usa variables no permitidas: {extras}. "
                     f"Solo puedes usar x1,...,x{n}.")
            st.stop()

        # ── Derivadas simbólicas y lambdify ────────────────────────────────────
        try:
            grad_sym = [sp.diff(f_sym, v) for v in variables]
            hess_sym = [[sp.diff(g, v) for v in variables] for g in grad_sym]
            f_lam     = sp.lambdify(variables, f_sym, 'numpy')
            grad_lams = [sp.lambdify(variables, g, 'numpy') for g in grad_sym]
            hess_lams = [[sp.lambdify(variables, h, 'numpy') for h in row] for row in hess_sym]
        except Exception as e:
            st.error(f"No se pudieron calcular las derivadas: {e}")
            st.stop()

        def f(x):
            return float(f_lam(*x))

        def grad(x):
            return np.array([float(g(*x)) for g in grad_lams])

        def hess(x):
            return np.array([[float(h(*x)) for h in row] for row in hess_lams])

        # ── Parseo del punto inicial ───────────────────────────────────────────
        partes = [v.strip() for v in punto_inicial.split(",")]
        if len(partes) != n:
            st.error(f"El punto de partida debe tener {n} valores. Tiene {len(partes)}.")
            st.stop()
        try:
            x0 = np.array([float(v) for v in partes])
        except ValueError as e:
            st.error(f"El punto de partida tiene valores no numéricos: {e}")
            st.stop()

        if not np.all(np.isfinite(x0)):
            st.error("El punto de partida contiene valores no finitos (NaN/Inf).")
            st.stop()

        # ── Validación de evaluación en x0 ─────────────────────────────────────
        try:
            v_test = f(x0)
            g_test = grad(x0)
            if not np.isfinite(v_test):
                st.error(f"f(x0) no es finito: {v_test}. Prueba con otro punto inicial.")
                st.stop()
            if not np.all(np.isfinite(g_test)):
                st.error("El gradiente en x0 no es finito. Prueba con otro punto inicial.")
                st.stop()
            if metodo == "Newton":
                H_test = hess(x0)
                if not np.all(np.isfinite(H_test)):
                    st.error("La Hessiana en x0 no es finita. Prueba con otro punto inicial.")
                    st.stop()
        except Exception as e:
            st.error(f"No se pudo evaluar la función o sus derivadas en x0: {e}")
            st.stop()

        # ── Búsqueda de línea con condiciones de Wolfe ─────────────────────────

        def zoom(alpha_lo, alpha_hi, x, p, phi0, dphi0):
            for _ in range(50):
                if abs(alpha_hi - alpha_lo) < 1e-12:
                    break
                alpha  = 0.5 * (alpha_lo + alpha_hi)
                phi_a  = f(x + alpha * p)
                phi_lo = f(x + alpha_lo * p)
                if phi_a > phi0 + c1_val * alpha * dphi0 or phi_a >= phi_lo:
                    alpha_hi = alpha
                else:
                    dphi_a = np.dot(grad(x + alpha * p), p)
                    if abs(dphi_a) <= -c2_val * dphi0:
                        return alpha
                    if dphi_a * (alpha_hi - alpha_lo) >= 0:
                        alpha_hi = alpha_lo
                    alpha_lo = alpha
            return 0.5 * (alpha_lo + alpha_hi)

        def wolfe_search(x, p):
            phi0  = f(x)
            dphi0 = np.dot(grad(x), p)
            if dphi0 >= 0:
                return 0.0
            alpha_max  = 10.0
            alpha_prev = 0.0
            phi_prev   = phi0
            alpha      = 1.0
            for i in range(50):
                phi_a = f(x + alpha * p)
                if phi_a > phi0 + c1_val * alpha * dphi0 or (i > 0 and phi_a >= phi_prev):
                    return zoom(alpha_prev, alpha, x, p, phi0, dphi0)
                dphi_a = np.dot(grad(x + alpha * p), p)
                if abs(dphi_a) <= -c2_val * dphi0:
                    return alpha
                if dphi_a >= 0:
                    return zoom(alpha, alpha_prev, x, p, phi0, dphi0)
                alpha_prev = alpha
                phi_prev   = phi_a
                alpha      = min(2 * alpha, alpha_max)
            return alpha

        def verificar_wolfe(x, p, alpha):
            phi0   = f(x)
            dphi0  = np.dot(grad(x), p)
            phi_a  = f(x + alpha * p)
            dphi_a = np.dot(grad(x + alpha * p), p)
            w1 = bool(phi_a <= phi0 + c1_val * alpha * dphi0)
            w2 = bool(abs(dphi_a) <= c2_val * abs(dphi0))
            return w1, w2

        ALPHA_EPS = 1e-15

        # ── Bucles de optimización ──────────────────────────────────────────────

        x           = x0.copy()
        historial   = []
        trayectoria = [x.copy()]
        alphas      = []
        wolfe1_vals = []
        wolfe2_vals = []
        razon_paro  = "máximo de iteraciones"

        if metodo == "Gradiente":
            for _ in range(max_it):
                g = grad(x)
                if not np.all(np.isfinite(g)):
                    razon_paro = "gradiente no finito"
                    break
                error = float(np.linalg.norm(g))
                historial.append(error)
                if error < tol:
                    razon_paro = "tolerancia alcanzada"
                    break
                p = -g
                alpha = wolfe_search(x, p)
                if alpha <= ALPHA_EPS:
                    razon_paro = "α ≈ 0 (búsqueda de línea no encontró paso válido)"
                    break
                w1, w2 = verificar_wolfe(x, p, alpha)
                alphas.append(alpha)
                wolfe1_vals.append(w1)
                wolfe2_vals.append(w2)
                x = x + alpha * p
                trayectoria.append(x.copy())

        elif metodo == "Gradiente Conjugado":
            g = grad(x)
            if not np.all(np.isfinite(g)):
                st.error("El gradiente inicial no es finito.")
                st.stop()
            p = -g
            for _ in range(max_it):
                error = float(np.linalg.norm(g))
                historial.append(error)
                if error < tol:
                    razon_paro = "tolerancia alcanzada"
                    break
                if np.dot(p, g) >= 0:
                    p = -g
                alpha = wolfe_search(x, p)
                if alpha <= ALPHA_EPS:
                    p = -g
                    alpha = wolfe_search(x, p)
                    if alpha <= ALPHA_EPS:
                        razon_paro = "α ≈ 0 (búsqueda de línea no encontró paso válido)"
                        break
                w1, w2 = verificar_wolfe(x, p, alpha)
                alphas.append(alpha)
                wolfe1_vals.append(w1)
                wolfe2_vals.append(w2)
                x_new = x + alpha * p
                g_new = grad(x_new)
                if not np.all(np.isfinite(g_new)):
                    trayectoria.append(x_new.copy())
                    razon_paro = "gradiente no finito"
                    break
                denom = np.dot(g, g)
                beta  = max(0.0, np.dot(g_new, g_new - g) / denom) if denom > 1e-20 else 0.0
                p     = -g_new + beta * p
                x, g  = x_new, g_new
                trayectoria.append(x.copy())

        elif metodo == "Newton":
            for _ in range(max_it):
                g = grad(x)
                if not np.all(np.isfinite(g)):
                    razon_paro = "gradiente no finito"
                    break
                error = float(np.linalg.norm(g))
                historial.append(error)
                if error < tol:
                    razon_paro = "tolerancia alcanzada"
                    break
                try:
                    H = hess(x)
                except Exception:
                    razon_paro = "no se pudo calcular la Hessiana"
                    break
                if not np.all(np.isfinite(H)):
                    p = -g
                else:
                    try:
                        np.linalg.cholesky(H)
                        p = -np.linalg.solve(H, g)
                        if not np.all(np.isfinite(p)) or np.dot(p, g) >= 0:
                            p = -g
                    except np.linalg.LinAlgError:
                        p = -g
                alpha = wolfe_search(x, p)
                if alpha <= ALPHA_EPS:
                    p = -g
                    alpha = wolfe_search(x, p)
                    if alpha <= ALPHA_EPS:
                        razon_paro = "α ≈ 0 (búsqueda de línea no encontró paso válido)"
                        break
                w1, w2 = verificar_wolfe(x, p, alpha)
                alphas.append(alpha)
                wolfe1_vals.append(w1)
                wolfe2_vals.append(w2)
                x = x + alpha * p
                trayectoria.append(x.copy())

        if len(historial) == 0:
            st.error("La optimización no pudo registrar ninguna iteración.")
            st.stop()

        # ── Resultados ─────────────────────────────────────────────────────────

        final_error = historial[-1]
        converged   = np.isfinite(final_error) and final_error < tol

        if converged:
            st.success(f"✅ Convergencia alcanzada en {len(historial)} iteraciones "
                       f"({razon_paro}).")
        else:
            st.warning(f"⚠️ No se alcanzó la tolerancia. "
                       f"Iteraciones: {len(historial)} · error final: {fmt(final_error)} · "
                       f"motivo: {razon_paro}.")

        if len(historial) <= 3:
            st.info(f"ℹ️ Solo {len(historial)} iteración(es): el punto inicial ya estaba "
                    f"cerca del mínimo o la función es muy simple (típico de Newton sobre cuadráticas).")

        try:
            f_final = f(x)
        except Exception:
            f_final = float('nan')

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Iteraciones",  len(historial))
        col_b.metric("f(x*)",         fmt(f_final))
        col_c.metric("‖∇f‖ final",    fmt(final_error, ".2e"))
        col_d.metric("Tolerancia",    f"{tol:.1e}")

        st.write(f"**Punto mínimo encontrado:** `{np.round(x, 8).tolist()}`")

        # ── Gráfico de convergencia ─────────────────────────────────────────────

        st.subheader("Convergencia")
        n_iters     = len(historial)
        use_markers = n_iters <= 80

        historial_log = [
            (max(h, LOG_FLOOR) if np.isfinite(h) else None)
            for h in historial
        ]
        historial_lin = [
            (h if np.isfinite(h) else None)
            for h in historial
        ]

        tab_log, tab_lin = st.tabs(["📉 Escala logarítmica", "📈 Escala lineal"])

        def make_conv_fig(y_data, log_scale):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(n_iters)),
                y=y_data,
                mode="lines+markers" if use_markers else "lines",
                name="‖∇f‖",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=5),
                connectgaps=False
            ))
            fig.add_hline(
                y=tol, line_dash="dash", line_color="red",
                annotation_text=f"Tolerancia {tol:.1e}",
                annotation_position="bottom right"
            )
            fig.update_layout(
                xaxis_title="Iteración",
                yaxis_title="‖∇f‖",
                yaxis_type="log" if log_scale else "linear",
                height=400
            )
            return fig

        with tab_log:
            st.plotly_chart(make_conv_fig(historial_log, log_scale=True),
                            use_container_width=True)
            st.caption("Una línea recta en escala log = convergencia lineal "
                       "(normal para Gradiente). Newton típicamente muestra una "
                       "caída abrupta = convergencia cuadrática.")

        with tab_lin:
            st.plotly_chart(make_conv_fig(historial_lin, log_scale=False),
                            use_container_width=True)

        # ── Paso α y condiciones de Wolfe ──────────────────────────────────────

        if len(alphas) > 0:
            st.subheader("Tamaño de paso α y condiciones de Wolfe")

            n_steps = len(alphas)
            n_w1    = sum(wolfe1_vals)
            n_w2    = sum(wolfe2_vals)
            n_both  = sum(a and b for a, b in zip(wolfe1_vals, wolfe2_vals))

            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Armijo (c1) ✅",    f"{n_w1}/{n_steps} pasos")
            wc2.metric("Curvatura (c2) ✅", f"{n_w2}/{n_steps} pasos")
            wc3.metric("Ambas condiciones", f"{n_both}/{n_steps} pasos")

            colors = [
                "green"  if (w1 and w2) else
                "orange" if w1          else
                "red"
                for w1, w2 in zip(wolfe1_vals, wolfe2_vals)
            ]

            alpha_min_pos = min((a for a in alphas if a > 0), default=1.0)
            alpha_range   = max(alphas) / (alpha_min_pos + 1e-15)

            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Scatter(
                x=list(range(n_steps)),
                y=alphas,
                mode="lines+markers",
                name="α",
                line=dict(color="#94a3b8", width=1),
                marker=dict(color=colors, size=10,
                            line=dict(color="black", width=0.5))
            ))
            fig_alpha.update_layout(
                xaxis_title="Iteración",
                yaxis_title="α",
                yaxis_type="log" if alpha_range > 100 else "linear",
                title="Paso α — 🟢 Wolfe OK | 🟠 Solo Armijo | 🔴 Ninguna condición",
                height=380
            )
            st.plotly_chart(fig_alpha, use_container_width=True)

            if n_steps <= 50:
                st.subheader("Detalle por iteración")
                table = {
                    "Iter":        list(range(1, n_steps + 1)),
                    "‖∇f‖":       [fmt(historial[i], ".4e") for i in range(n_steps)],
                    "α":           [fmt(a, ".6f") for a in alphas],
                    "Armijo ✓":    ["✅" if w else "❌" for w in wolfe1_vals],
                    "Curvatura ✓": ["✅" if w else "❌" for w in wolfe2_vals],
                }
                st.dataframe(table, use_container_width=True)

        # ── Visualización geométrica ────────────────────────────────────────────

        tray = np.array(trayectoria)

        def eval_grid(X, Y):
            """Evalúa f vectorizado si se puede, si no, elemento por elemento."""
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
                            Z[i, j] = f([X[i, j], Y[i, j]])
                        except Exception:
                            pass
                return Z

        if n == 1:
            st.subheader("Función y trayectoria (1D)")
            tray_1d = tray[:, 0]
            rango = tray_1d.max() - tray_1d.min()
            pad = max(1.0, rango * 0.5)
            xs = np.linspace(tray_1d.min() - pad, tray_1d.max() + pad, 300)

            ys = np.full_like(xs, np.nan)
            try:
                ys_try = f_lam(xs)
                ys_arr = np.asarray(ys_try, dtype=float)
                if ys_arr.shape == xs.shape:
                    ys = ys_arr
                else:
                    ys = np.full_like(xs, float(ys_try))
            except Exception:
                for i, xv in enumerate(xs):
                    try:
                        ys[i] = f([xv])
                    except Exception:
                        pass

            tray_y = []
            for xv in tray_1d:
                try:
                    tray_y.append(f([xv]))
                except Exception:
                    tray_y.append(np.nan)

            fig1d = go.Figure()
            fig1d.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name="f(x1)",
                line=dict(color="#2563eb", width=2)
            ))
            fig1d.add_trace(go.Scatter(
                x=tray_1d, y=tray_y,
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=8, color="red"),
                name="Trayectoria"
            ))
            fig1d.add_trace(go.Scatter(
                x=[tray_1d[0]], y=[tray_y[0]],
                mode="markers",
                marker=dict(size=16, color="yellow", symbol="star"),
                name="Inicio"
            ))
            fig1d.add_trace(go.Scatter(
                x=[tray_1d[-1]], y=[tray_y[-1]],
                mode="markers",
                marker=dict(size=16, color="lime", symbol="star"),
                name="Mínimo"
            ))
            fig1d.update_layout(xaxis_title="x1", yaxis_title="f(x1)", height=500)
            st.plotly_chart(fig1d, use_container_width=True)

        elif n == 2:
            st.subheader("Trayectoria sobre curvas de nivel")

            rango_x = tray[:, 0].max() - tray[:, 0].min()
            rango_y = tray[:, 1].max() - tray[:, 1].min()
            pad_x   = max(1.0, rango_x * 0.5)
            pad_y   = max(1.0, rango_y * 0.5)

            xx = np.linspace(tray[:, 0].min() - pad_x, tray[:, 0].max() + pad_x, 80)
            yy = np.linspace(tray[:, 1].min() - pad_y, tray[:, 1].max() + pad_y, 80)
            X, Y = np.meshgrid(xx, yy)
            Z = eval_grid(X, Y)

            z_finite = Z[np.isfinite(Z)]
            if z_finite.size == 0:
                st.warning("No se pudo calcular el mapa de curvas de nivel "
                           "(todos los valores son no finitos en la región mostrada).")
            else:
                z_low  = np.percentile(z_finite, 2)
                z_high = np.percentile(z_finite, 98)
                if z_low == z_high:
                    z_low  -= 1.0
                    z_high += 1.0
                Z_clip = np.clip(Z, z_low, z_high)

                fig2 = go.Figure()
                fig2.add_trace(go.Contour(
                    x=xx, y=yy, z=Z_clip,
                    colorscale="Viridis",
                    contours=dict(showlabels=True, coloring="heatmap"),
                    name="f(x1,x2)"
                ))
                if len(tray) > 1:
                    fig2.add_trace(go.Scatter(
                        x=tray[:, 0], y=tray[:, 1],
                        mode="lines+markers",
                        line=dict(color="red", width=2),
                        marker=dict(size=6, color="red"),
                        name="Trayectoria"
                    ))
                fig2.add_trace(go.Scatter(
                    x=[tray[0, 0]], y=[tray[0, 1]],
                    mode="markers",
                    marker=dict(size=16, color="yellow", symbol="star"),
                    name="Inicio"
                ))
                fig2.add_trace(go.Scatter(
                    x=[tray[-1, 0]], y=[tray[-1, 1]],
                    mode="markers",
                    marker=dict(size=16, color="lime", symbol="star"),
                    name="Mínimo"
                ))
                fig2.update_layout(xaxis_title="x1", yaxis_title="x2", height=600)
                st.plotly_chart(fig2, use_container_width=True)

        else:  # n > 2
            st.subheader("Evolución de variables durante la optimización")
            fig_vars = go.Figure()
            for k in range(n):
                fig_vars.add_trace(go.Scatter(
                    x=list(range(len(tray))),
                    y=tray[:, k],
                    mode="lines+markers" if len(tray) <= 80 else "lines",
                    name=f"x{k+1}"
                ))
            fig_vars.update_layout(
                xaxis_title="Iteración",
                yaxis_title="Valor de variable",
                height=400
            )
            st.plotly_chart(fig_vars, use_container_width=True)
            st.caption("ℹ️ El mapa de curvas de nivel solo está disponible para n = 2 variables.")

    except Exception as e:
        import traceback
        st.error(f"Error inesperado: {e}")
        with st.expander("Ver detalle del error"):
            st.code(traceback.format_exc())
