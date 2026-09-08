# ============================================================
# python_efarm_par.py
# Κεφάλαιο 12 — Εφαρμογές Παραγώγου
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
import numpy as np
from sympy import symbols, diff, series, limit, exp, sin, cos, log, oo, solve

x, l = symbols('x l', positive=True)

print("=" * 55)
print(" Κεφάλαιο 12: Εφαρμογές Παραγώγου")
print("=" * 55)

# 1. Σειρά Maclaurin
print("\n[1] Σειρές Maclaurin")
for name, f_, n_ in [("e^x",exp(x),6),("sin(x)",sin(x),7),("cos(x)",cos(x),6),
                      ("ln(1+x)",log(1+x),6),("1/(1-x)",1/(1-x),5)]:
    print(f"  {name:12s} ≈ {series(f_,x,0,n_).removeO()}")

# 2. L'Hopital
print("\n[2] Κανόνας L'Hopital")
x = symbols('x')
for desc, expr, pt in [
    ("sin(x)/x", sin(x)/x, 0),
    ("(e^x-1)/x",(exp(x)-1)/x, 0),
    ("x*e^(-x)", x*exp(-x), oo),
    ("(1-cos(x))/x^2",(1-cos(x))/x**2, 0)]:
    print(f"  lim x->{pt} {desc} = {limit(expr,x,pt)}")

# 3. Βελτιστοποίηση
print("\n[3] Βελτιστοποίηση — μέγιστο εμβαδόν πλαισίου (περίμ.=20)")
l2 = symbols('l', positive=True)
A = l2*(10-l2)
l_opt = solve(diff(A,l2), l2)[0]
print(f"  βέλτιστο l={l_opt},  A_max={A.subs(l2,l_opt)}")

# 4. Newton
print("\n[4] Μέθοδος Newton — ρίζα x^3=2")
from sympy import lambdify
f_n = x**3 - 2
f_n_num = lambdify(x, f_n, 'numpy')
df_n_num = lambdify(x, diff(f_n,x), 'numpy')
xi = 1.5
for i in range(8):
    fxi = f_n_num(xi)
    print(f"  i={i}  x={xi:.8f}  f(x)={fxi:.2e}")
    if abs(fxi)<1e-12: break
    xi -= fxi/df_n_num(xi)
print("\n✓ Ολοκληρώθηκε.")

# ============================================================
# ΣΥΜΠΛΗΡΩΜΑ — Newton & βελτιστοποίηση με SciPy
#   scipy.optimize.newton, scipy.optimize.minimize_scalar,
#   np.linspace, matplotlib.pyplot
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize_scalar

# --- Μέθοδος Newton: ρίζα της f(x) = x^3 - 2x - 5 ---
f  = lambda x: x**3 - 2*x - 5
df = lambda x: 3*x**2 - 2

r = newton(f, x0=2.0, fprime=df, tol=1e-12, full_output=True)
print("Ρίζα (Newton) =", r[0])
print("Επαναλήψεις   =", r[1].iterations, " f(ρίζα) =", f(r[0]))

# Χωρίς παράγωγο (τέμνουσα):
print("Ρίζα (τέμνουσα) =", newton(f, x0=2.0, tol=1e-12))

# --- Βελτιστοποίηση: ελάχιστο της g(x) = x^2 - 4x + 7 ---
g = lambda x: x**2 - 4*x + 7
res = minimize_scalar(g, bounds=(-10, 10), method='bounded')
print(f"\nΕλάχιστο της g στο x = {res.x:.6f}, g(x) = {res.fun:.6f}")
print("Αναλυτικά: x = 2, g(2) = 3")

# --- Γραφική επιβεβαίωση ---
xs = np.linspace(-1, 5, 400)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(xs, f(xs), lw=2); ax[0].axhline(0, color='k', lw=.6)
ax[0].plot(r[0], 0, 'ro'); ax[0].set_title("f(x)=x³-2x-5 και η ρίζα της")
ax[0].grid(alpha=.3)
ax[1].plot(xs, g(xs), lw=2, color='seagreen')
ax[1].plot(res.x, res.fun, 'ro'); ax[1].set_title("g(x)=x²-4x+7 και το ελάχιστο")
ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig('ch12_newton_opt.png', dpi=100)
print("\nΓράφημα: ch12_newton_opt.png")
