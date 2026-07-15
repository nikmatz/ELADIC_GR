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
