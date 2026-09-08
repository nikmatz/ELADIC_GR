# python_diafex.py — Διαφορικές Εξισώσεις (Τόμος 2)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# Απαιτεί: pip install sympy matplotlib numpy scipy

from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

x, t = symbols('x t', real=True)
y = Function('y')
C1, C2 = symbols('C1 C2')

print("=" * 60)
print("ΔΙΑΦΟΡΙΚΕΣ ΕΞΙΣΩΣΕΙΣ (ΔΕ)")
print("=" * 60)

# ── 1. Διαχωρίσιμη ΔΕ ───────────────────────────────────────────────────────
print()
print("1. ΔΙΑΧΩΡΙΣΙΜΗ ΔΕ:  y' = y·x")
ode1 = Eq(y(x).diff(x), y(x)*x)
sol1 = dsolve(ode1, y(x))
print(f"   Γενική λύση: {sol1}")

# Αρχική συνθήκη y(0)=2
C_val = solve(sol1.rhs.subs(x, 0) - 2, symbols('C1'))[0]
sol1_part = sol1.rhs.subs(symbols('C1'), C_val)
print(f"   Μερική λύση (y(0)=2): y = {sol1_part}")

# ── 2. Γραμμική 1ης τάξης ───────────────────────────────────────────────────
print()
print("2. ΓΡΑΜΜΙΚΗ ΔΕ 1ης ΤΑΞΗΣ:  y' + 2y = 4x")
ode2 = Eq(y(x).diff(x) + 2*y(x), 4*x)
sol2 = dsolve(ode2, y(x))
print(f"   Γενική λύση: {sol2}")

# ── 3. Ομογενής 2ης τάξης με σταθερούς συντελεστές ─────────────────────────
print()
print("3. ΟΜΟΓΕΝΗΣ 2ης ΤΑΞΗΣ:  y'' - 3y' + 2y = 0")
ode3 = Eq(y(x).diff(x, 2) - 3*y(x).diff(x) + 2*y(x), 0)
sol3 = dsolve(ode3, y(x))
print(f"   Γενική λύση: {sol3}")

# ── 4. Μη-ομογενής — Μέθοδος Αόριστης Συνιστάμενης ────────────────────────
print()
print("4. ΜΗ-ΟΜΟΓΕΝΗΣ:  y'' + y = sin(x)")
ode4 = Eq(y(x).diff(x, 2) + y(x), sin(x))
sol4 = dsolve(ode4, y(x))
print(f"   Γενική λύση: {sol4}")

# ── 5. Πεδίο κλίσεων (slope field) ──────────────────────────────────────────
print()
print("5. ΠΕΔΙΟ ΚΛΙΣΕΩΝ:  y' = x - y")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Slope field
xv = np.linspace(-2, 4, 20)
yv = np.linspace(-2, 4, 20)
X, Y = np.meshgrid(xv, yv)
dydx = X - Y
norm = np.sqrt(1 + dydx**2)
axes[0].quiver(X, Y, 1/norm, dydx/norm, alpha=0.6, color='gray')

# Μερικές λύσεις (για διάφορες αρχικές συνθήκες)
t_span = (xv.min(), xv.max())
t_eval = np.linspace(*t_span, 200)
for y0 in [-1, 0, 1, 2, 3]:
    sol = solve_ivp(lambda t, y: t - y[0], t_span, [y0], t_eval=t_eval, method='RK45')
    axes[0].plot(sol.t, sol.y[0], 'b-', linewidth=1.5, alpha=0.8)
axes[0].set_xlim(-2, 4); axes[0].set_ylim(-2, 4)
axes[0].set_title("Πεδίο κλίσεων: y' = x - y")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].grid(True, alpha=0.2)

# Euler vs Ακριβής
print()
print("6. ΜΕΘΟΔΟΣ EULER:  y' = -2y,  y(0)=1")
def euler_method(f, x0, y0, h, n_steps):
    xs, ys = [x0], [y0]
    for _ in range(n_steps):
        y0 = y0 + h * f(xs[-1], y0)
        x0 = x0 + h
        xs.append(x0); ys.append(y0)
    return np.array(xs), np.array(ys)

f_euler = lambda xi, yi: -2*yi
x_exact = np.linspace(0, 3, 300)
y_exact = np.exp(-2*x_exact)

for h, color, label in [(0.5, 'r', 'h=0.5'), (0.2, 'g', 'h=0.2'), (0.1, 'm', 'h=0.1')]:
    n_steps = int(3/h)
    xe, ye = euler_method(f_euler, 0, 1, h, n_steps)
    axes[1].plot(xe, ye, f'{color}o--', markersize=4, label=f'Euler {label}')
axes[1].plot(x_exact, y_exact, 'b-', linewidth=2, label='Ακριβής $e^{-2x}$')
axes[1].set_title("Μέθοδος Euler: y' = -2y")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig('diafex.png', dpi=100); plt.show()
print("Αρχείο 'diafex.png' αποθηκεύτηκε.")
