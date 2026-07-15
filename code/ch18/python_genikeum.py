# python_genikeum.py — Γενικευμένα Ολοκληρώματα (Κεφ. 18)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# Απαιτεί: pip install sympy matplotlib

from sympy import *
import matplotlib.pyplot as plt
import numpy as np

x = symbols('x', real=True)
t = symbols('t', positive=True)

print("=" * 60)
print("ΓΕΝΙΚΕΥΜΕΝΑ ΟΛΟΚΛΗΡΩΜΑΤΑ (Improper Integrals)")
print("=" * 60)

# ── 1. Ολοκλήρωμα σε άπειρο διάστημα ────────────────────────────────────────
print()
print("1. ∫_1^∞  1/x²  dx")
I1 = integrate(1/x**2, (x, 1, oo))
print(f"   = {I1}  (σύγκλιση)")

print()
print("2. ∫_1^∞  1/x  dx")
I2 = integrate(1/x, (x, 1, oo))
print(f"   = {I2}  (απόκλιση)")

# ── 2. Ολοκλήρωμα με ιδιαίτερο σημείο ──────────────────────────────────────
print()
print("3. ∫_0^1  1/√x  dx  (ιδιαίτερο σημείο x=0)")
I3 = integrate(1/sqrt(x), (x, 0, 1))
print(f"   = {I3}  (σύγκλιση)")

print()
print("4. ∫_0^1  1/x  dx  (ιδιαίτερο σημείο x=0)")
I4 = integrate(1/x, (x, 0, 1))
print(f"   = {I4}  (απόκλιση)")

# ── 3. Κριτήριο σύγκλισης p-ολοκληρώματος ───────────────────────────────────
print()
print("5. Κριτήριο p — ∫_1^∞  1/xᵖ  dx:")
p = symbols('p', real=True)
for pval in [Rational(1,2), 1, Rational(3,2), 2, 3]:
    result = integrate(1/x**pval, (x, 1, oo))
    status = "σύγκλιση" if result.is_finite else "απόκλιση"
    print(f"   p = {pval}: {result}  ({status})")

# ── 4. Εκθετικό ολοκλήρωμα (σχετικό με Γ-συνάρτηση) ─────────────────────────
print()
print("6. ∫_0^∞  e^{-x}  dx  =", integrate(exp(-x), (x, 0, oo)))
print("7. ∫_0^∞  x·e^{-x}  dx  =", integrate(x*exp(-x), (x, 0, oo)))
print("8. ∫_0^∞  x²·e^{-x}  dx  =", integrate(x**2*exp(-x), (x, 0, oo)))

# ── 5. Γράφημα σύγκλισης ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
xv = np.linspace(1, 10, 500)
axes[0].plot(xv, 1/xv**2, 'b-', label=r'$1/x^2$ (σύγκλιση)')
axes[0].fill_between(xv, 1/xv**2, alpha=0.3, color='blue')
axes[0].plot(xv, 1/xv, 'r-', label=r'$1/x$ (απόκλιση)')
axes[0].set_ylim(0, 2); axes[0].legend(); axes[0].set_title(r'Σύγκριση $1/x$ και $1/x^2$')
axes[0].grid(True, alpha=0.3)

xv2 = np.linspace(0.01, 1, 500)
axes[1].plot(xv2, 1/np.sqrt(xv2), 'g-', label=r'$1/\sqrt{x}$ (σύγκλιση)')
axes[1].fill_between(xv2, 1/np.sqrt(xv2), alpha=0.3, color='green')
axes[1].plot(xv2, 1/xv2, 'r-', label=r'$1/x$ (απόκλιση)')
axes[1].set_ylim(0, 20); axes[1].legend()
axes[1].set_title(r'Ιδιαίτερο σημείο $x=0$'); axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig('improper.png', dpi=100); plt.show()
print()
print("Αρχείο 'improper.png' αποθηκεύτηκε.")
