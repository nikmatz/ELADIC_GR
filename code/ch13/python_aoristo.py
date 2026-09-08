# ============================================================
# python_aoristo.py
# Κεφάλαιο 13 — Αόριστο Ολοκλήρωμα
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.integrate(f, x)   -> αόριστο ολοκλήρωμα
#   sympy.apart(f, x)       -> μερικά κλάσματα
#   sympy.diff(result, x)   -> επαλήθευση
# ============================================================
import sympy as sp

x = sp.symbols('x')
print("=" * 55)
print(" Κεφάλαιο 13: Αόριστο Ολοκλήρωμα")
print("=" * 55)

# 1. Βασικά
print("\n[1] Βασικοί τύποι")
for name, f_ in [("x^5",x**5),("1/x",1/x),("e^x",sp.exp(x)),
                  ("sin(x)",sp.sin(x)),("cos(x)",sp.cos(x)),
                  ("1/sqrt(1-x^2)",1/sp.sqrt(1-x**2)),
                  ("1/(1+x^2)",1/(1+x**2))]:
    print(f"  int({name:18s}) = {sp.integrate(f_,x)} + C")

# 2. Αντικατάσταση
print("\n[2] Αντικατάσταση u=g(x)")
for name, f_ in [("sin(2x+1)",sp.sin(2*x+1)),
                  ("x*e^(x^2)",x*sp.exp(x**2)),
                  ("x/sqrt(x^2+4)",x/sp.sqrt(x**2+4)),
                  ("cos^3(x)*sin(x)",sp.cos(x)**3*sp.sin(x))]:
    print(f"  int({name:22s}) = {sp.simplify(sp.integrate(f_,x))} + C")

# 3. Κατά μέρη
print("\n[3] Κατά μέρη: int(u dv) = uv - int(v du)")
for name, f_ in [("x*e^x",x*sp.exp(x)),("x*sin(x)",x*sp.sin(x)),
                  ("x^2*ln(x)",x**2*sp.log(x)),("ln(x)",sp.log(x))]:
    print(f"  int({name:15s}) = {sp.integrate(f_,x)} + C")

# 4. Επαλήθευση
print("\n[4] Επαλήθευση")
f_test = x**3 * sp.exp(x)
I = sp.integrate(f_test, x)
chk = sp.simplify(sp.diff(I,x) - f_test) == 0
print(f"  d/dx[int(x^3*e^x)] = x^3*e^x ?  {chk}")
print("\n✓ Ολοκληρώθηκε.")

# ============================================================
# ΣΥΜΠΛΗΡΩΜΑ — Η οικογένεια αντιπαραγώγων F(x)+c
#   sympy.lambdify, matplotlib.pyplot
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x = sp.symbols('x')
f = 3*x**2 - 2*x
F = sp.integrate(f, x)
print("f(x) =", f, "   F(x) =", F, "+ c")
print("Έλεγχος: F'(x) =", sp.simplify(sp.diff(F, x)))

F_num = sp.lambdify(x, F, 'numpy')
f_num = sp.lambdify(x, f, 'numpy')
xs = np.linspace(-1.5, 2.5, 400)

plt.figure(figsize=(7, 4.5))
for c in (-3, -1, 0, 1, 3):
    plt.plot(xs, F_num(xs) + c, lw=1.6, label=f"F(x) + ({c})")
plt.axhline(0, color='k', lw=.6); plt.axvline(0, color='k', lw=.6)
plt.title("Οικογένεια αντιπαραγώγων: κατακόρυφες μετατοπίσεις της ίδιας καμπύλης")
plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig('ch13_antiparagogoi.png', dpi=100)
print("Γράφημα: ch13_antiparagogoi.png")

# Η κλίση είναι ίδια για κάθε c — αυτό ακριβώς σημαίνει «+ c»:
for c in (-3, 0, 3):
    print(f"  c={c:+d}: κλίση στο x=1 -> {float(sp.diff(F + c, x).subs(x, 1))}"
          f"   (f(1) = {float(f.subs(x, 1))})")
