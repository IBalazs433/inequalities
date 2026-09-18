# Linear Inequalities Visualizer

A small, portable Streamlit app for visualizing systems of linear
inequalities in two variables (x, y). Type inequalities like
`2x + 3y <= 12`, and instantly see the shaded feasible region, boundary
lines, and corner points ("vertices").

## Setup

Requires Python 3.9+.

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

Streamlit will open the app in your browser (usually at
http://localhost:8501). Close the terminal or press `Ctrl+C` to stop it.

## Sharing with colleagues

Just send them this folder (`app.py`, `requirements.txt`, and this
`README.md`) — no notebook or extra setup needed. Anyone with Python
installed can run it with the three commands above.

## How to use

- **Inequalities** box: one per line, e.g.
  - `x + y <= 6`
  - `2x - y < 4`
  - `y >= 0`
- **Lines** box (optional): equations that are drawn but not shaded, e.g.
  `y = 2x + 1` — handy for objective-function lines in linear programming.
- Adjust the x/y range, tick/grid spacing, colors, and opacities in the
  side panel; the plot updates live.
- Toggle "Show vertices" to mark and label the corner points of the
  feasible region (where all inequalities hold simultaneously).

## Example

The app opens pre-loaded with a small linear-programming-style example:

```
x + y <= 6
2x + y <= 8
x >= 0
y >= 0
```

This shows a feasible region bounded by four constraints, with its
corner points labeled — a good starting point to demonstrate the concept
before students try their own systems.
