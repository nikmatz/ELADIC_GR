# ============================================================
# python_synartiseis.py
# Κεφάλαιο 8 — Συναρτήσεις Πραγματικής Μεταβλητής
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy     -> συμβολικοί υπολογισμοί, πεδίο ορισμού
#   numpy     -> αριθμητική αποτίμηση
#   matplotlib -> γράφημα συνάρτησης
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.lambdify(x, expr)  -> μετατροπή σε αριθμητική f(x)
#   sympy.solve(expr, x)     -> μηδενικά / πεδίο ορισμού
#   sympy.simplify()         -> απλοποίηση
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sqrt, log, Abs, lambdify, solve, simplify, pprint

x = symbols('x')

print("=" * 58)
print(" Κεφάλαιο 8: Συναρτήσεις Πραγματικής Μεταβλητής")
print("=" * 58)

# 1. Ορισμός και αποτίμηση
print("\n[1] Αποτίμηση συνάρτησης f(x) = x^3 - 3x")
f_expr = x**3 - 3*x
f_num  = lambdify(x, f_expr, 'numpy')
for xi in [-2, 0, 1, 2]:
    print(f"  f({xi}) = {f_num(xi)}")

# 2. Πεδίο ορισμού
print("\n[2] Πεδίο ορισμού (ανισώσεις)")
from sympy import S, solveset, Reals
funcs = [
    ("sqrt(4 - x^2)",     sqrt(4 - x**2)),
    ("log(x - 1)",        log(x - 1)),
    ("1 / (x^2 - x - 2)", 1 / (x**2 - x - 2)),
]
for name, expr in funcs:
    try:
        dom = solveset(expr, x, domain=S.Reals)
        print(f"  {name:28s} => {dom}")
    except Exception:
        print(f"  {name:28s} => (χειροκίνητος υπολογισμός)")

# 3. Σύνθεση
print("\n[3] Σύνθεση f o g")
f1 = x**2 + 1
g1 = 2*x - 3
fog = f1.subs(x, g1)
gof = g1.subs(x, f1)
print(f"  f(x)={f1},  g(x)={g1}")
print(f"  (fog)(x) = {simplify(fog)}")
print(f"  (gof)(x) = {simplify(gof)}")

# 4. Αντίστροφη
print("\n[4] Αντίστροφη f(x)=2x+5")
y = symbols('y')
inv = solve(2*x + 5 - y, x)[0]
print(f"  f^-1(y) = {inv}  =>  f^-1(x) = {inv.subs(y,x)}")

# 5. Άρτια / Περιττή
print("\n[5] Άρτια / Περιττή")
test_funcs = [x**4 - 2*x**2, x**3 - x, x**2 + x]
for f_ in test_funcs:
    even = simplify(f_.subs(x,-x) - f_) == 0
    odd  = simplify(f_.subs(x,-x) + f_) == 0
    kind = "Άρτια" if even else ("Περιττή" if odd else "Ούτε")
    print(f"  f(x)={str(f_):20s}  -> {kind}")

# 6. Γράφημα
xs = np.linspace(-2.5, 2.5, 400)
plt.figure(figsize=(6,4))
plt.plot(xs, f_num(xs), 'b-', lw=2, label=r'$f(x)=x^3-3x$')
plt.axhline(0, color='k', lw=0.5, ls='--')
plt.axvline(0, color='k', lw=0.5, ls='--')
plt.xlabel('x'); plt.grid(True, alpha=0.3); plt.legend()
plt.title('f(x) = x³ - 3x'); plt.tight_layout()
plt.savefig('ch08_graph.png', dpi=100)
print("\n  Γράφημα: ch08_graph.png")
print("\n✓ Ολοκληρώθηκε.")
