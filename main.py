import streamlit as st
import numpy as np
from scipy.optimize import linprog

# Function to solve the linear programming problem
def solve_lp(num_vars, num_constraints, optimization_goal, objective, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq):
    if optimization_goal == "Maximize":
        objective = [-x for x in objective]

    # Convert constraints to the appropriate form for linprog
    lhs_ineq = np.array(lhs_ineq) if lhs_ineq else None
    rhs_ineq = np.array(rhs_ineq) if rhs_ineq else None
    lhs_eq = np.array(lhs_eq) if lhs_eq else None
    rhs_eq = np.array(rhs_eq) if lhs_eq else None

    # Solve the linear program
    res = linprog(
        c=objective,
        A_ub=lhs_ineq,
        b_ub=rhs_ineq,
        A_eq=lhs_eq,
        b_eq=rhs_eq,
        method='highs',
    )

    if res.success:
        result_str = "Optimal solution:\n"
        for i, value in enumerate(res.x):
            result_str += f"x{i + 1} = {value:.4f}\n"

        optimal_value = -res.fun if optimization_goal == "Maximize" else res.fun
        result_str += f"Optimal value of the objective function: {optimal_value:.4f}"
        return result_str
    else:
        return "Couldn't find an optimal solution. Problem might be infeasible or unbounded."

# Streamlit interface
st.title("Simplex Method Calculator")

num_vars = st.number_input("Number of variables:", min_value=1, value=2)
num_constraints = st.number_input("Number of constraints:", min_value=1, value=2)

optimization_goal = st.selectbox("Optimization goal:", ["Minimize", "Maximize"])

# Objective function section
st.subheader("Objective Function")

cols_per_row = 4  # Number of variables per row for better readability
objective = []
objective_chunks = [list(range(i, min(i + cols_per_row, num_vars))) for i in range(0, num_vars, cols_per_row)]

for chunk in objective_chunks:
    if len(chunk) > 0:  # Ensure chunk is not empty
        cols = st.columns(len(chunk))
        for i, col in enumerate(cols):
            with col:
                coeff = st.number_input(f"x{chunk[i] + 1} coefficient:", value=0.0, key=f"obj_coeff_{chunk[i]}")
                objective.append(coeff)

# Constraints section
st.subheader("Constraints")

lhs_ineq = []
rhs_ineq = []
lhs_eq = []
rhs_eq = []

for i in range(num_constraints):
    with st.expander(f"Constraint {i + 1}"):
        cols = st.columns(cols_per_row + 2)  # Include columns for inequality sign and RHS value
        lhs = []
        for j in range(num_vars):
            with cols[j % cols_per_row]:  # Wrap rows for better readability
                coeff = st.number_input(f"x{j + 1} (Constraint {i + 1})", value=0.0, key=f"lhs_{i}_{j}")
                lhs.append(coeff)

        with cols[-2]:  # Second last column for inequality sign
            inequality_sign = st.selectbox(f"Inequality {i + 1}", ["<=", ">=", "="], key=f"sign_{i}")

        with cols[-1]:  # Last column for RHS
            rhs = st.number_input(f"RHS for constraint {i + 1}:", value=0.0, key=f"rhs_{i}")

        if inequality_sign == "<=":
            lhs_ineq.append(lhs)
            rhs_ineq.append(rhs)
        elif inequality_sign == ">=":
            lhs_ineq.append([-x for x in lhs])
            rhs_ineq.append(-rhs)
        else:
            lhs_eq.append(lhs)
            rhs_eq.append(rhs)

if st.button("Solve"):
    solution = solve_lp(num_vars, num_constraints, optimization_goal, objective, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq)
    st.subheader("Solution")
    st.text(solution)

"""
test comment
"""

"""
Testing github codespaces
"""