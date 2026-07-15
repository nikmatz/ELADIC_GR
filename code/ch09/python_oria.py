# ============================================================
# python_oria.py
# Κεφάλαιο 9 — Όρια και Συνέχεια
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy   -> συμβολικά όρια: limit(f, x, a)
#   numpy   -> αριθμητική προσέγγιση
#   matplotlib -> γράφημα συνέχειας / ασυνέχειας
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.limit(f, x, a)       -> lim_{x->a} f(x)
#   sympy.limit(f, x, a, '+')  -> δεξιό όριο
#   sympy.limit(f, x, a, '-')  -> αριστερό όριο
#   sympy.limit(f, x, oo)      -> όριο στο άπειρο
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, limit, oo, sin, cos, exp, log, Abs, Piecewise, simplify

x = symbols('x')

print("=" * 55)
print(" Κεφάλαιο 9: Όρια και Συνέχεια")
print("=" * 55)

# 1. Βασικά όρια
print("\n[1] Υπολογισμός ορίων")
cases = [
    ("lim x->2  (x^2-4)/(x-2)",   (x**2-4)/(x-2),    2),
    ("lim x->0  sin(x)/x",          sin(x)/x,           0),
    ("lim x->inf (2x^2+1)/(x^2-3)", (2*x**2+1)/(x**2-3), oo),
    ("lim x->0  (e^x-1)/x",         (exp(x)-1)/x,       0),
    ("lim x->0  (1-cos(x))/x^2",    (1-cos(x))/x**2,    0),
    ("lim x->inf (1+1/x)^x",        (1+1/x)**x,         oo),
]
for desc, expr, pt in cases:
    print(f"  {desc:42s} = {limit(expr, x, pt)}")

# 2. Μονόπλευρα
print("\n[2] Μονόπλευρα όρια")
print(f"  lim x->0+ |x|/x = {limit(Abs(x)/x, x, 0, '+')}")
print(f"  lim x->0- |x|/x = {limit(Abs(x)/x, x, 0, '-')}")
print(f"  lim x->1+ log(x-1) = {limit(log(x-1), x, 1, '+')}")

# 3. Αριθμητική προσέγγιση
print("\n[3] Αριθμητική προσέγγιση: sin(x)/x -> 1")
for xi in [0.5, 0.1, 0.01, 0.001, 0.0001]:
    print(f"  x={xi:.4f}  sin(x)/x = {np.sin(xi)/xi:.10f}")

# 4. Συνέχεια
print("\n[4] Έλεγχος συνέχειας f(x)=sin(x)/x στο x=0")
lim_val = limit(sin(x)/x, x, 0)
f_at_0  = 1  # ορίζεται έτσι
print(f"  lim x->0 f(x) = {lim_val},  f(0) = {f_at_0}  -> Συνεχής: {lim_val == f_at_0}")

# 5. Θεώρημα Bolzano
print("\n[5] Θεώρημα Bolzano: f(x)=x^3-x-2 στο [1,2]")
from sympy import lambdify
f_expr = x**3 - x - 2
f_num  = lambdify(x, f_expr, 'numpy')
print(f"  f(1) = {f_num(1)},  f(2) = {f_num(2)}  -> Ρίζα στο (1,2): {f_num(1)*f_num(2) < 0}")
# bisection
a_b, b_b = 1.0, 2.0
for _ in range(40):
    m = (a_b+b_b)/2
    if f_num(m)*f_num(a_b) < 0: b_b = m
    else: a_b = m
print(f"  Προσεγγ. ρίζα: {m:.8f}")

# 6. Γράφημα
xs = np.linspace(-4*np.pi, 4*np.pi, 800)
ys = np.where(np.abs(xs) < 1e-10, 1.0, np.sin(xs)/xs)
plt.figure(figsize=(8,4))
plt.plot(xs, ys, 'b-', lw=2)
plt.scatter([0],[1], color='red', zorder=5, s=50)
plt.axhline(0,color='k',lw=0.5); plt.grid(True,alpha=0.3)
plt.title(r'$\sin(x)/x$ — αφαιρετή ασυνέχεια')
plt.tight_layout(); plt.savefig('ch08_sinc.png', dpi=100)
print("\n  Γράφημα: ch08_sinc.png")
print("\n✓ Ολοκληρώθηκε.")
