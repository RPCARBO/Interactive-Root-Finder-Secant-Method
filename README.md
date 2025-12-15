# Interactive Root-Finder: Secant Method
This repository contains an interactive Streamlit application for exploring the Secant Method in numerical root‑finding. The app is designed to be both educational and practical, offering a calculator‑style interface for building mathematical functions and a clear visualization of the iterative process.

## Features
Function Builder UI: Clickable buttons (Addition, Subtraction, Multiplication, Division, sin, cos, exp, etc.) let users construct functions without typing raw code.

Implicit Multiplication Support: Expressions like 5x are automatically interpreted as 5*x.

Tolerance Selector: Choose from a range of very coarse to very fine tolerance values using a discrete slider.

Iteration Table: Displays each step of the Secant Method with values of Initial Guess 1, Initial Guess 2, the next approximation, the function value, and error.

Graphical Visualization: Plots the function and highlights the approximate root.

Accessible Controls: Word‑labeled operator buttons (Addition, Subtraction, Multiplication, Division) for clarity and usability.

## Tech Stack
Python

Streamlit for the interactive UI

SymPy for safe mathematical parsing

NumPy & Pandas for numerical operations and tabular data

Matplotlib for plotting

## Getting Started
Install dependencies:

pip install streamlit sympy numpy pandas matplotlib

Run the app:

streamlit run secant_app.py
