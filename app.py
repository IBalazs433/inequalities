"""
Linear Inequalities Visualizer
==============================

A self-contained Streamlit app for exploring systems of linear inequalities
in two variables (x, y). Type inequalities like `2x + 3y <= 12`, and the app
shades the feasible region, draws the boundary lines, and marks the corner
points (vertices) of the feasible region -- exactly what you need when
studying linear programming / systems of inequalities.

Run it with:
    streamlit run app.py
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------------------
# Core math: parsing inequalities and plotting them
# (ported from the original notebook function, unchanged in behavior)
# --------------------------------------------------------------------------

_OPS = ["<=", ">=", "==", "<", ">", "="]


def _coeffs(expr):
    """Turn a linear expression in x, y into (a, b, c) with a*x + b*y + c."""
    e = expr.replace("^", "**")
    e = re.sub(r"(?<=[\d\)])\s*(?=[xy\(])", "*", e)   # 2x -> 2*x, )(  -> )*(
    e = re.sub(r"(?<=[xy])\s+(?=[\d\(])", "*", e)     # x 2 -> x*2

    def f(x, y):
        return float(eval(e, {"__builtins__": {}}, {"x": x, "y": y}))

    c = f(0.0, 0.0)
    return f(1.0, 0.0) - c, f(0.0, 1.0) - c, c


def parse(s):
    """'2x + 3y <= 12' -> (a, b, c, op) meaning a*x + b*y + c <op> 0."""
    if not isinstance(s, str):          # already (a, b, c) or (a, b, c, op)
        a, b, c, *rest = s
        return a, b, c, (rest[0] if rest else "<=")

    for op in _OPS:
        if op in s:
            lhs, rhs = s.split(op, 1)
            a1, b1, c1 = _coeffs(lhs)
            a2, b2, c2 = _coeffs(rhs)
            a, b, c = a1 - a2, b1 - b2, c1 - c2
            if op in (">=", ">"):                       # flip to <= / <
                a, b, c = -a, -b, -c
                op = "<=" if op == ">=" else "<"
            if op == "==":
                op = "="
            return a, b, c, op

    raise ValueError(f"No comparison operator in {s!r}")


def plot_inequalities(
    inequalities,
    lines=(),
    xlim=(-10, 10),
    ylim=(-10, 10),
    n=500,
    alpha=0.15,
    intersection_alpha=0.45,
    color="tab:blue",
    line_color="tab:red",
    figsize=(6, 6),
    grid_step=1,
    tick_step=2,
    show_vertices=True,
):
    """
    inequalities: strings like "x + y <= 6", "y >= 0", "2x - y < 4"
    lines:        strings like "y = 2x + 1" (drawn only, not shaded)

    Equations passed inside `inequalities` are treated as lines too.

    Returns the matplotlib Figure and the list of feasible-region vertices.
    """
    parsed = [parse(s) for s in inequalities] + [
        (*parse(s)[:3], "=") for s in lines
    ]
    constraints = [p for p in parsed if p[3] != "="]
    equations = [p for p in parsed if p[3] == "="]

    x = np.linspace(*xlim, n)
    y = np.linspace(*ylim, n)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=figsize)
    rgb = plt.matplotlib.colors.to_rgb(color)

    def shade(mask, a):
        rgba = np.zeros((*X.shape, 4))
        rgba[..., :3] = rgb
        rgba[..., 3] = a * mask
        ax.imshow(rgba, extent=(*xlim, *ylim), origin="lower", aspect="equal")

    intersection = np.ones(X.shape, dtype=bool)

    for a, b, c, op in constraints:
        F = a * X + b * Y + c
        region = F < 0 if op == "<" else F <= 0
        intersection &= region
        shade(region, alpha)
        ax.contour(
            X, Y, F,
            levels=[0],
            colors="black",
            linewidths=1.5,
            linestyles="dashed" if op == "<" else "solid",
        )

    if constraints:
        shade(intersection, intersection_alpha)

    for a, b, c, _ in equations:
        ax.contour(X, Y, a * X + b * Y + c, levels=[0],
                   colors=line_color, linewidths=1.8)

    # Extreme points: intersections of boundaries that satisfy every inequality
    vertices = []
    if show_vertices:
        tol = 1e-9
        for i, (a1, b1, c1, _) in enumerate(parsed):
            for a2, b2, c2, _ in parsed[i + 1:]:
                A = np.array([[a1, b1], [a2, b2]])
                if abs(np.linalg.det(A)) < 1e-10:
                    continue
                p = np.linalg.solve(A, [-c1, -c2])
                if not (xlim[0] - tol <= p[0] <= xlim[1] + tol
                        and ylim[0] - tol <= p[1] <= ylim[1] + tol):
                    continue
                if all(a * p[0] + b * p[1] + c <= tol
                       for a, b, c, _ in constraints):
                    if not any(np.linalg.norm(p - q) < 1e-7 for q in vertices):
                        vertices.append(p)

        for x_v, y_v in vertices:
            ax.scatter(x_v, y_v, color="black", s=35, zorder=5)
            ax.annotate(f"({x_v:g}, {y_v:g})", (x_v, y_v),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=10, color="black", zorder=6)

    ax.axhline(0, color="gray", lw=1.0)
    ax.axvline(0, color="gray", lw=1.0)

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(tick_step))
        axis.set_minor_locator(MultipleLocator(grid_step))

    ax.grid(which="major", alpha=0.35)
    ax.grid(which="minor", alpha=0.15)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    return fig, vertices


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Linear Inequalities Visualizer", layout="wide")

st.title("📐 Linear Inequalities Visualizer")
st.markdown(
    "Explore systems of linear inequalities in two variables. Enter one "
    "inequality per line and see the feasible region, boundary lines, "
    "and corner points update instantly."
)

with st.expander("ℹ️ How to write inequalities and lines", expanded=False):
    st.markdown(
        """
**Inequalities** describe a shaded region. Write one per line, using `x` and `y`
as the variables. Supported operators: `<=`, `>=`, `<`, `>`, `=` (or `==`).

Examples:
- `x + y <= 6`
- `2x - y < 4`
- `y >= 0`
- `3(x + 1) <= 2y - 5`

**Lines** are drawn (in a different color) but *not* shaded — useful for
objective-function lines (e.g. in linear programming) or reference lines.
Write them as equations, one per line:
- `y = 2x + 1`
- `x + y = 10`

Notes:
- Multiplication can be implicit: `2x` means `2*x`, and `3(x+1)` means `3*(x+1)`.
- Powers use `^` or `**` (though this tool is meant for *linear* expressions).
- The feasible region (where **all** inequalities hold at once) is shaded
  darker than the individual regions.
- Corner points ("vertices") of the feasible region are marked automatically.
        """
    )

col_input, col_plot = st.columns([1, 1.4], gap="large")

with col_input:
    st.subheader("System of inequalities")
    inequalities_text = st.text_area(
        "One inequality per line",
        value="2x + y <= 100\nx + y <= 80\nx <= 40\nx >= 0\ny >= 0",
        height=140,
        key="inequalities",
    )

    st.subheader("Extra lines (optional)")
    lines_text = st.text_area(
        "One equation per line (drawn only, not shaded)",
        value="",
        height=70,
        key="lines",
        placeholder="e.g. y = 2x + 1",
    )

    st.subheader("View settings")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**x range**")
        xc1, xc2 = st.columns(2)
        xlim_min = xc1.number_input("x min", value=-10, step=1, key="xlim_min")
        xlim_max = xc2.number_input("x max", value=90, step=1, key="xlim_max")
    with c2:
        st.markdown("**y range**")
        yc1, yc2 = st.columns(2)
        ylim_min = yc1.number_input("y min", value=-10, step=1, key="ylim_min")
        ylim_max = yc2.number_input("y max", value=90, step=1, key="ylim_max")

    if xlim_min >= xlim_max:
        st.warning("x min must be less than x max; using defaults (-10, 90).")
        xlim_min, xlim_max = -10, 90
    if ylim_min >= ylim_max:
        st.warning("y min must be less than y max; using defaults (-10, 90).")
        ylim_min, ylim_max = -10, 90

    c3, c4, c5 = st.columns(3)
    with c3:
        tick_step = st.number_input("Tick step", min_value=1, value=10, step=1)
    with c4:
        grid_step = st.number_input("Grid step", min_value=1, value=10, step=1)
    with c5:
        show_vertices = st.checkbox("Show vertices", value=True)

    c6, c7 = st.columns(2)
    with c6:
        shade_color = st.color_picker("Shading color", value="#e60101")
    with c7:
        line_color = st.color_picker("Lines color", value="#1445e9")

    alpha = st.slider("Individual region opacity", 0.0, 1.0, 0.1, 0.05)
    intersection_alpha = st.slider(
        "Feasible region opacity", 0.0, 1.0, 0.7, 0.05
    )

with col_plot:
    st.subheader("Plot")

    inequalities = [
        line.strip() for line in inequalities_text.splitlines() if line.strip()
    ]
    lines = [line.strip() for line in lines_text.splitlines() if line.strip()]

    if not inequalities and not lines:
        st.info("Enter at least one inequality or line to see a plot.")
    else:
        try:
            fig, vertices = plot_inequalities(
                inequalities,
                lines=lines,
                xlim=(xlim_min, xlim_max),
                ylim=(ylim_min, ylim_max),
                alpha=alpha,
                intersection_alpha=intersection_alpha,
                color=shade_color,
                line_color=line_color,
                grid_step=grid_step,
                tick_step=tick_step,
                show_vertices=show_vertices,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            if show_vertices and vertices:
                st.caption("Feasible region vertices:")
                st.write(
                    ", ".join(f"({v[0]:g}, {v[1]:g})" for v in vertices)
                )
        except Exception as exc:
            st.error(f"Couldn't parse your input: {exc}")

st.divider()
st.caption(
    "Built with Streamlit and Matplotlib. Share this folder (app.py + "
    "requirements.txt) with anyone — run `pip install -r requirements.txt` "
    "then `streamlit run app.py`."
)
