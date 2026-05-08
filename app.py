import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go

st.title("Optimizador de Funciones")
st.subheader("Parámetros de entrada")

funcion = st.text_input("Función objetivo (usa x1, x2, etc.)", value="x1**2 + x2**2")
n_vars = st.number_input("Número de variables", min_value=1, max_value=5, value=2)
metodo = st.selectbox("Método de optimización", ["Gradiente", "Gradiente Conjugado", "Newton"])
punto_inicial = st.text_input("Punto de partida (separado por comas)", value="2, 2")
max_iter = st.number_input("Máximo de iteraciones", min_value=10, max_value=10000, value=200)
tolerancia = st.number_input("Tolerancia de convergencia", value=1e-6, format="%.2e")
c1 = st.number_input("Parámetro Wolfe c1 (Armijo)", value=1e-4, format="%.4f")
c2 = st.number_input("Parámetro Wolfe c2 (curvatura)", value=0.9, format="%.2f")

if st.button("Optimizar"):
    try:
        n = int(n_vars)
        c1_val = float(c1)
        c2_val = float(c2)
        tol = float(tolerancia)
        max_it = int(max_iter)

        if not (0 < c1_val < c2_val < 1):
            st.error(f"Los parámetros Wolfe deben cumplir 0 < c1 < c2 < 1. Tienes c1={c1_val}, c2={c2_val}.")
            st.stop()

        variables = sp.symbols(f'x1:{n+1}')
        f_sym = sp.sympify(funcion)

        extras = f_sym.free_symbols - set(variables)
        if extras:
            st.error(f"La función usa variables no permitidas: {extras}. Solo puedes usar x1,...,x{n}.")
            st.stop()

        grad_sym = [sp.diff(f_sym, v) for v in variables]
        hess_sym = [[sp.diff(g, v) for v in variables] for g in grad_sym]

        f_lam = sp.lambdify(variables, f_sym, 'numpy')
        grad_lams = [sp.lambdify(variables, g, 'numpy') for g in grad_sym]
        hess_lams = [[sp.lambdify(variables, h, 'numpy') for h in row] for row in hess_sym]

        def f(x):
            return float(f_lam(*x))

        def grad(x):
            return np.array([float(g(*x)) for g in grad_lams])

        def hess(x):
            return np.array([[float(h(*x)) for h in row] for row in hess_lams])

        partes = [v.strip() for v in punto_inicial.split(",")]
        if len(partes) != n:
            st.error(f"El punto de partida debe tener {n} valores. Tiene {len(partes)}.")
            st.stop()
        x0 = np.array([float(v) for v in partes])

        try:
            _ = f(x0); _ = grad(x0)
            if metodo == "Newton":
                _ = hess(x0)
        except Exception as e:
            st.error(f"No se pudo evaluar la función o sus derivadas: {e}")
            st.stop()

        def zoom(alpha_lo, alpha_hi, x, p, phi0, dphi0):
            alpha = 0.5 * (alpha_lo + alpha_hi)
            for _ in range(50):
                if abs(alpha_hi - alpha_lo) < 1e-12:
                    return alpha
                alpha = 0.5 * (alpha_lo + alpha_hi)
                phi_a = f(x + alpha * p)
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
            return alpha

        def wolfe_search(x, p):
            phi0 = f(x)
            dphi0 = np.dot(grad(x), p)
            if dphi0 >= 0:
                return 0.0
            alpha_max = 10.0
            alpha_prev = 0.0
            phi_prev = phi0
            alpha = 1.0
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
                phi_prev = phi_a
                alpha = min(2 * alpha, alpha_max)
            return alpha

        x = x0.copy()
        historial = []
        trayectoria = [x.copy()]

        if metodo == "Gradiente":
            for i in range(max_it):
                g = grad(x)
                error = float(np.linalg.norm(g))
                historial.append(error)
                if error < tol:
                    break
                p = -g
                alpha = wolfe_search(x, p)
                if alpha == 0:
                    break
                x = x + alpha * p
                trayectoria.append(x.copy())

        elif metodo == "Gradiente Conjugado":
            g = grad(x)
            p = -g
            for i in range(max_it):
                error = float(np.linalg.norm(g))
                historial.append(error)
                if error < tol:
                    break
                if np.dot(p, g) >= 0:
                    p = -g
                alpha = wolfe_search(x, p)
                if alpha == 0:
                    p = -g
                    alpha = wolfe_search(x, p)
                    if alpha == 0:
                        break
                x_new = x + alpha * p
                g_new = grad(x_new)
                denom = np.dot(g, g)
                beta = max(0.0, np.dot(g_new, g_new - g) / denom) if denom > 1e-20 else 0.0
                p = -g_new + beta * p
                x, g = x_new, g_new
                trayectoria.append(x.copy())

        elif metodo == "Newton":
            for i in range(max_it):
                g = grad(x)
                error = float(np.linalg.norm(g))
                historial.append(error)
                if error < tol:
                    break
                H = hess(x)
                p = None
                try:
                    np.linalg.cholesky(H)
                    p = -np.linalg.solve(H, g)
                    if np.dot(p, g) >= 0:
                        p = -g
                except np.linalg.LinAlgError:
                    p = -g
                alpha = wolfe_search(x, p)
                if alpha == 0:
                    p = -g
                    alpha = wolfe_search(x, p)
                    if alpha == 0:
                        break
                x = x + alpha * p
                trayectoria.append(x.copy())

        if len(historial) == 0:
            st.error("La optimización no pudo iniciar.")
            st.stop()

        if historial[-1] < tol:
            st.success(f"¡Convergencia alcanzada en {len(historial)} iteraciones!")
        else:
            st.warning(f"No se alcanzó la tolerancia. Iteraciones: {len(historial)}, error final: {historial[-1]:.2e}")

        st.write(f"**Punto mínimo:** {x}")
        st.write(f"**Valor de la función:** {f(x):.6e}")
        st.write(f"**Iteraciones realizadas:** {len(historial)}")
        st.write(f"**Error final (||∇f||):** {historial[-1]:.2e}")

        st.subheader("Convergencia")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=historial, mode='lines+markers', name='||∇f||'))
        fig.update_layout(xaxis_title='Iteración', yaxis_title='||∇f||', yaxis_type='log')
        st.plotly_chart(fig)

        if n == 2 and len(trayectoria) >= 1:
            st.subheader("Trayectoria de optimización")
            tray = np.array(trayectoria)
            rango_x = max(1.0, tray[:,0].max() - tray[:,0].min())
            rango_y = max(1.0, tray[:,1].max() - tray[:,1].min())
            mx = 0.4 * rango_x + 0.5
            my = 0.4 * rango_y + 0.5
            xx = np.linspace(tray[:,0].min() - mx, tray[:,0].max() + mx, 80)
            yy = np.linspace(tray[:,1].min() - my, tray[:,1].max() + my, 80)
            X, Y = np.meshgrid(xx, yy)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        Z[i, j] = f([X[i, j], Y[i, j]])
                    except Exception:
                        Z[i, j] = np.nan
            fig2 = go.Figure()
            fig2.add_trace(go.Contour(x=xx, y=yy, z=Z, colorscale='Viridis',
                                       contours=dict(showlabels=True), name='f(x1,x2)'))
            fig2.add_trace(go.Scatter(x=tray[:,0], y=tray[:,1], mode='lines+markers',
                                       line=dict(color='red', width=2),
                                       marker=dict(size=8, color='red'), name='Trayectoria'))
            fig2.add_trace(go.Scatter(x=[tray[0,0]], y=[tray[0,1]], mode='markers',
                                       marker=dict(size=18, color='yellow', symbol='star'),
                                       name='Inicio'))
            fig2.add_trace(go.Scatter(x=[tray[-1,0]], y=[tray[-1,1]], mode='markers',
                                       marker=dict(size=18, color='lime', symbol='star'),
                                       name='Mínimo'))
            fig2.update_layout(xaxis_title='x1', yaxis_title='x2', height=600)
            st.plotly_chart(fig2)

    except Exception as e:
        st.error(f"Error: {e}")
