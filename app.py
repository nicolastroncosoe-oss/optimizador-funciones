import streamlit as st
import numpy as np

st.title("Optimizador de Funciones")
st.subheader("Parámetros de entrada")

funcion = st.text_input("Función objetivo (usa x1, x2, etc.)", value="x1**2 + x2**2")

n_vars = st.number_input("Número de variables", min_value=1, max_value=5, value=2)

metodo = st.selectbox("Método de optimización", [
    "Gradiente",
    "Gradiente Conjugado", 
    "Newton"
])

punto_inicial = st.text_input("Punto de partida (separado por comas)", value="2, 2")

max_iter = st.number_input("Máximo de iteraciones", min_value=10, max_value=10000, value=100)

tolerancia = st.number_input("Tolerancia de convergencia", value=1e-6, format="%.2e")

c1 = st.number_input("Parámetro Wolfe c1 (Armijo)", value=1e-4, format="%.4f")
c2 = st.number_input("Parámetro Wolfe c2 (curvatura)", value=0.9, format="%.2f")

st.button("Optimizar")
