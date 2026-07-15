# python_efarm_ol.py — Εφαρμογές Ολοκληρωτικού Λογισμού (Κεφ. 17)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# Απαιτεί: pip install sympy matplotlib numpy

from sympy import *
import numpy as np
import matplotlib.pyplot as plt

x = symbols('x', real=True)

# ── 1. Εμβαδόν μεταξύ δύο καμπυλών ──────────────────────────────────────────
print("=" * 60)
print("1. ΕΜΒΑΔΟΝ ΜΕΤΑΞΥ ΔΥΟ ΚΑΜΠΥΛΩΝ")
print("=" * 60)

f = x**2
g = x + 2
# Τομή
pts = solve(f - g, x)
a, b = min(pts), max(pts)
print(f"Καμπύλες: f(x)={f},  g(x)={g}")
print(f"Τομές: x = {a}, x = {b}")
A = integrate(g - f, (x, a, b))
print(f"Εμβαδόν = ∫_({a})^({b}) (g-f) dx = {A}")

# Γράφημα
xv = np.linspace(float(a)-0.5, float(b)+0.5, 300)
fv = np.array([float(f.subs(x, xi)) for xi in xv])
gv = np.array([float(g.subs(x, xi)) for xi in xv])
plt.figure(figsize=(7, 4))
plt.plot(xv, fv, 'b-', label=r'$f(x)=x^2$')
plt.plot(xv, gv, 'r-', label=r'$g(x)=x+2$')
plt.fill_between(xv, fv, gv, where=(gv >= fv), alpha=0.25, color='green', label=f'Εμβαδόν = {A}')
plt.axhline(0, color='k', linewidth=0.5)
plt.legend(); plt.title('Εμβαδόν μεταξύ δύο καμπυλών'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('area_between.png', dpi=100); plt.show()

# ── 2. Όγκος Περιστροφής (μέθοδος δίσκων) ───────────────────────────────────
print()
print("=" * 60)
print("2. ΟΓΚΟΣ ΠΕΡΙΣΤΡΟΦΗΣ ΓΥΡΩ ΑΠΟ ΤΟΝ x-ΑΞΟΝΑ")
print("=" * 60)

f2 = sqrt(x)
a2, b2 = 0, 4
V = pi * integrate(f2**2, (x, a2, b2))
print(f"f(x) = {f2},  [a, b] = [{a2}, {b2}]")
print(f"V = π ∫_0^4 f(x)² dx = {V} ≈ {float(V):.4f}")

# ── 3. Μήκος Καμπύλης ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. ΜΗΚΟΣ ΚΑΜΠΥΛΗΣ")
print("=" * 60)

f3 = x**Rational(3, 2)
a3, b3 = 0, 4
df3 = diff(f3, x)
L = integrate(sqrt(1 + df3**2), (x, a3, b3))
print(f"f(x) = {f3},  [a, b] = [{a3}, {b3}]")
print(f"L = ∫_0^4 √(1 + f'²) dx = {simplify(L)}")

# ── 4. Εφαρμογή: Ροπή Αδράνειας ─────────────────────────────────────────────
print()
print("=" * 60)
print("4. ΕΦΑΡΜΟΓΗ: ΡΟΠΗ ΑΔΡΑΝΕΙΑΣ ΟΜΟΙΟΓΕΝΟΥΣ ΕΛΑΣΤΡΟΥ")
print("=" * 60)

rho = symbols('rho', positive=True)   # γραμμική πυκνότητα
f4 = 1  # ομοιογενές έλαστρο
a4, b4 = 0, symbols('L', positive=True)
m = rho * integrate(f4, (x, a4, b4))
I = rho * integrate(x**2 * f4, (x, a4, b4))
print(f"Μάζα m = ρ·L = {m}")
print(f"Ροπή αδράνειας Ι = ρ ∫x² dx = {I}  (= mL²/3)")

print()
print("Αρχείο 'area_between.png' αποθηκεύτηκε.")
