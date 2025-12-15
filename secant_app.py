import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
import re

st.set_page_config(page_title="Secant Method Root Finder", layout="centered")
st.title("Secant Method Root Finder")

# ---------------------------
# Session state initialization
# ---------------------------
def init_state():
    defaults = {
        "expr": "",
        "x0": 1.0,
        "x1": 2.0,
        "max_iter": 100,
        "tol": 1e-6
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------------------
# Expression helpers
# ---------------------------
def add_token(token: str):
    st.session_state.expr += token

def backspace():
    st.session_state.expr = st.session_state.expr[:-1] if st.session_state.expr else ""

def clear_expr():
    st.session_state.expr = ""

def parse_f(x):
    try:
        expr = parse_expr(
            st.session_state.expr,
            transformations=standard_transformations + (implicit_multiplication_application,)
        )
        return float(expr.subs(sp.Symbol("x"), x))
    except Exception as e:
        st.error(f"Function error: {e}")
        return None

# ---------------------------
# UI: Function builder
# ---------------------------
st.subheader("Build your function f(x)")
st.caption("Press buttons to construct your function. Example: sin(x) + 5x - 2")

st.text_input("Function being built", key="expr", disabled=True)
 
# Organized calculator-style grid for the function builder
grid = [
    ["sin(", "cos(", "exp(", "(", ")"],
    ["^", "x", "/", "⌫", "C"],
    ["7", "8", "9", "Multiply", "Sub"],
    ["4", "5", "6", "Add", "."],
    ["1", "2", "3", "0", " "]
]

# Map display labels to tokens used by add_token (keep most labels as-is)
display_to_token = {
    "^": "**",
    "⌫": "BACK",
    "C": "CLEAR",
    "Multiply": "*",
    "Add": "+",
    "Sub": "-"
}

for r_idx, row in enumerate(grid):
    cols = st.columns(len(row))
    for c_idx, label in enumerate(row):
        # empty placeholder cell
        if label.strip() == "":
            cols[c_idx].write("")
            continue

        # sanitize label for use in the widget key to avoid issues with special chars
        safe_label = re.sub(r'[^0-9A-Za-z]', lambda m: f"_{ord(m.group(0))}_", label)
        key = f"btn_{r_idx}_{c_idx}_{safe_label}"
        if label == "⌫":
            cols[c_idx].button("⌫", key=key, on_click=backspace)
        elif label == "C":
            cols[c_idx].button("C", key=key, on_click=clear_expr)
        else:
            token = display_to_token.get(label, label)
            cols[c_idx].button(label, key=key, on_click=add_token, args=(token,))

# ---------------------------
# Sidebar parameters
# ---------------------------
st.sidebar.header("Parameters")
st.session_state.x0 = st.sidebar.number_input("Initial guess x₀", value=float(st.session_state.x0))
st.session_state.x1 = st.sidebar.number_input("Initial guess x₁", value=float(st.session_state.x1))
st.session_state.max_iter = st.sidebar.number_input("Max iterations", value=int(st.session_state.max_iter), step=1)

# ---------------------------
# Tolerance (discrete selector 10^-1 to 10^-8)
# ---------------------------
st.subheader("Tolerance")
tol_options = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
# Keep current value if valid, else default to 1e-6
current_tol = st.session_state.tol if st.session_state.tol in tol_options else 1e-6
tol_value = st.select_slider(
    "Select tolerance",
    options=tol_options,
    value=current_tol,
    format_func=lambda v: f"{v:.0e}"
)
st.session_state.tol = tol_value

# ---------------------------
# Secant method
# ---------------------------
def secant_method(x0, x1, tol, max_iter):
    data = []
    x_prev, x_curr = x0, x1
    for i in range(1, max_iter + 1):
        f_prev = parse_f(x_prev)
        f_curr = parse_f(x_curr)
        if f_prev is None or f_curr is None:
            return None, data
        denom = (f_curr - f_prev)
        if denom == 0:
            st.warning("Division by zero encountered in secant step.")
            return None, data
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        error = abs(x_next - x_curr)
        f_next = parse_f(x_next)
        data.append({
            "Iteration": i,
            "x₀": x_prev,
            "x₁": x_curr,
            "x₂": x_next,
            "f(x₂)": f_next,
            "Error": error
        })
        if error < tol:
            return x_next, data
        x_prev, x_curr = x_curr, x_next
    return None, data

# ---------------------------
# Run
# ---------------------------
if st.sidebar.button("Run Secant Method", width="stretch"):
    root, results = secant_method(st.session_state.x0, st.session_state.x1, st.session_state.tol, st.session_state.max_iter)
    if root is not None:
        st.success(f"Approximate root found: {root:.6f}")
        df = pd.DataFrame(results)
        st.subheader("Iteration table")
        st.dataframe(
            df.style.format({
                "x₀": "{:.6f}",
                "x₁": "{:.6f}",
                "x₂": "{:.6f}",
                "f(x₂)": "{:.6f}",
                "Error": "{:.2e}"
            }),
            width="stretch"
        )

        st.subheader("Function plot")
        lo = min(st.session_state.x0, st.session_state.x1) - 2
        hi = max(st.session_state.x0, st.session_state.x1) + 2
        x_vals = np.linspace(lo, hi, 600)
        y_vals = []
        for xv in x_vals:
            val = parse_f(xv)
            y_vals.append(np.nan if val is None else val)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="f(x)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(root, color="red", linestyle="--", linewidth=1, label=f"Root ≈ {root:.6f}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Root not found. Check your function or initial guesses.")
