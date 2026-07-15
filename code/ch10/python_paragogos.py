# ============================================================
# python_paragogos.py
# Κεφάλαιο 10 — Παράγωγος Συνάρτησης
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy -> diff(f,x), diff(f,x,n), idiff()
#   matplotlib -> γράφημα εφαπτομένης
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.diff(f, x)       -> f'(x)
#   sympy.diff(f, x, n)    -> n-οστή παράγωγος
#   sympy.idiff(F, y, x)   -> άρρητη: F(x,y)=0
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, sin, cos, tan, exp, log, sqrt, idiff, lambdify, simplify

x, y = symbols('x y')

print("=" * 55)
print(" Κεφάλαιο 10: Παράγωγος Συνάρτησης")
print("=" * 55)

# 1. Βασικοί κανόνες
print("\n[1] Κανόνες παραγώγισης")
examples = [
    ("x^5 - 3x^2 + 7",  x**5 - 3*x**2 + 7),
    ("sin(x)*cos(x)",    sin(x)*cos(x)),
    ("x^2 * e^x",        x**2*exp(x)),
    ("ln(x)/x",          log(x)/x),
    ("(x^2+1)^10",       (x**2+1)**10),
    ("sin(x^2)",         sin(x**2)),
    ("e^sin(x)",         exp(sin(x))),
]
for name, expr in examples:
    print(f"  ({name})' = {simplify(diff(expr,x))}")

# 2. Ανώτερης τάξης
print("\n[2] Ανώτερης τάξης: f(x)=sin(x)")
for n in range(1,5):
    print(f"  f^({n})(x) = {diff(sin(x),x,n)}")

# 3. Κανόνας αλυσίδας
print("\n[3] Κανόνας αλυσίδας (chain rule)")
for name, expr in [("sin(3x^2+1)",sin(3*x**2+1)),("e^(x^2-x)",exp(x**2-x)),
                   ("ln(cos(x))",log(cos(x))),("sqrt(x^2+4)",sqrt(x**2+4))]:
    print(f"  ({name})' = {simplify(diff(expr,x))}")

# 4. Άρρητη παραγώγιση
print("\n[4] Άρρητη: x^2+y^2=25  -> dy/dx=?")
eq = x**2 + y**2 - 25
dydx = idiff(eq, y, x)
print(f"  dy/dx = {dydx}  (= -x/y)")

# 5. Εφαπτόμενη
print("\n[5] Εφαπτόμενη f(x)=x^3 στο x0=1")
f_e = x**3
df_e = diff(f_e,x)
x0, y0 = 1.0, 1.0
slope = float(df_e.subs(x,x0))
f_num  = lambdify(x, f_e, 'numpy')
tang   = lambda t: y0 + slope*(t-x0)

xs = np.linspace(-1.5,2.5,300)
plt.figure(figsize=(6,4))
plt.plot(xs, f_num(xs),'b-',lw=2,label=r'$x^3$')
plt.plot(xs, tang(xs), 'r--',lw=1.5,label=f'εφαπτόμενη x0=1')
plt.scatter([x0],[y0],color='red',s=60,zorder=5)
plt.xlim(-1.5,2.5); plt.ylim(-3,6)
plt.grid(True,alpha=0.3); plt.legend()
plt.title('Εφαπτόμενη γραμμή'); plt.tight_layout()
plt.savefig('ch09_tangent.png',dpi=100)
print(f"  κλίση={slope}  Γράφημα: ch09_tangent.png")
print("\n✓ Ολοκληρώθηκε.")
