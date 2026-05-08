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
max_iter = st.number_input("Máximo de iteraciones", min_value=10, max_value=10000, value=100)
tolerancia = st.number_input("Tolerancia de convergencia", value=1e-6, format="%.2e")
c1 = st.number_input("Parámetro Wolfe c1 (Armijo)", value=1e-4, format="%.4f")
c2 = st.number_input("Parámetro Wolfe c2 (curvatura)", value=0.9, format="%.2f")

if st.button("Optimizar"):
    try:
        variables = [sp.Symbol(f'x{i+1}') for i in range(int(n_vars))]
        f_sym = sp.sympify(funcion)
        grad_sym = [sp.diff(f_sym, v) for v in variables]
        hess_sym = [[sp.diff(g, v) for v in variables] for g in grad_sym]

        f = lambda x: float(f_sym.subs(zip(variables, x)))
        grad = lambda x: np.array([float(g.subs(zip(variables, x))) for g in grad_sym])
        hess = lambda x: np.array([[float(h.subs(zip(variables, x))) for h in row] for row in hess_sym])

        x0 = np.array([float(v) for v in punto_inicial.split(",")])

        def wolfe_search(x, p, f, grad, c1, c2):
            alpha, a, b = 1.0, 0.0, float('inf')
            for _ in range(100):
                if f(x + alpha*p) > f(x) + c1*alpha*np.dot(grad(x), p):
                    b = alpha
                    alpha = (a + b) / 2
                elif np.dot(grad(x + alpha*p), p) < c2*np.dot(grad(x), p):
                    a = alpha
                    alpha = min(2*a, (a+b)/2) if b == float('inf') else (a+b)/2
                else:
                    break
            return alpha

        x = x0.copy()
        historial = []
        tol = float(tolerancia)

        if metodo == "Gradiente":
            for i in range(int(max_iter)):
                g = grad(x)
                error = np.linalg.norm(g)
                historial.append(error)
                if error < tol:
                    break
                p = -g
                alpha = wolfe_search(x, p, f, grad, c1, c2)
                x = x + alpha * p

        elif metodo == "Gradiente Conjugado":
            g = grad(x)
            p = -g
            for i in range(int(max_iter)):
                error = np.linalg.norm(g)
                historial.append(error)
                if error < tol:
                    break
                alpha = wolfe_search(x, p, f, grad, c1, c2)
                x_new = x + alpha * p
                g_new = grad(x_new)
                beta = max(0, np.dot(g_new, g_new - g) / np.dot(g, g))
                p = -g_new + beta * p
                x, g = x_new, g_new

        elif metodo == "Newton":
            for i in range(int(max_iter)):
                g = grad(x)
                error = np.linalg.norm(g)
                historial.append(error)
                if error < tol:
                    break
                H = hess(x)
                try:
                    p = -np.linalg.solve(H, g)
                except:
                    p = -g
                alpha = wolfe_search(x, p, f, grad, c1, c2)
                x = x + alpha * p

        st.success("¡Optimización completada!")
        st.write(f"**Punto mínimo:** {x}")
        st.write(f"**Valor de la función:** {f(x):.6f}")
        st.write(f"**Iteraciones realizadas:** {len(historial)}")
        st.write(f"**Error final:** {historial[-1]:.2e}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=historial, mode='lines', name='Error'))
        fig.update_layout(title='Convergencia', xaxis_title='Iteración', yaxis_title='Error', yaxis_type='log')
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error: {e}")
