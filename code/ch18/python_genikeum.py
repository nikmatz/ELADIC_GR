# python_genikeum.py — Γενικευμένα Ολοκληρώματα (Κεφ. 18)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# Εντολές: sp.integrate(f,(x,1,sp.oo)), scipy.integrate.quad, scipy.special.gamma, sp.oo

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import gamma as Gamma, beta as Beta
import math

x = sp.symbols('x', positive=True)

print("=" * 60)
print("ΓΕΝΙΚΕΥΜΕΝΑ ΟΛΟΚΛΗΡΩΜΑΤΑ (improper integrals)")
print("=" * 60)

# ── Α. Το κριτήριο του p για τα ολοκληρώματα Α' είδους ───────────────────────
print()
print("Α. ∫_1^∞ x^(-p) dx  —  συμβολικά με SymPy")
print("-" * 60)
for p in [sp.Rational(1, 2), sp.Integer(1), sp.Integer(2)]:
    I = sp.integrate(x**(-p), (x, 1, sp.oo))
    status = "σύγκλιση" if I.is_finite else "απόκλιση"
    print(f"   p = {str(p):3s} :  ∫ = {str(I):12s}  ({status})")
print("   Συμπέρασμα: συγκλίνει ΑΚΡΙΒΩΣ όταν p > 1.")
print("   (αναμενόμενα: p=1/2 -> oo, p=1 -> oo, p=2 -> 1)")

# ── Β. Το ολοκλήρωμα Gauss αριθμητικά ───────────────────────────────────────
print()
print("Β. ∫_0^∞ e^(-x²) dx  —  αριθμητικά με scipy.integrate.quad")
print("-" * 60)
val, err = quad(lambda t: np.exp(-t**2), 0, np.inf)
exact = np.sqrt(np.pi) / 2
print(f"   quad          = {val:.15f}   (εκτ. σφάλμα {err:.1e})")
print(f"   √π/2          = {exact:.15f}")
print(f"   |διαφορά|     = {abs(val - exact):.3e}   (μέσα στην ακρίβεια διπλής λέξης)")
print(f"   SymPy συμβολικά: {sp.integrate(sp.exp(-x**2), (x, 0, sp.oo))}")

# ── Γ. Η συνάρτηση Γάμμα ────────────────────────────────────────────────────
print()
print("Γ. Η ΣΥΝΑΡΤΗΣΗ ΓΑΜΜΑ:  Γ(s) = ∫_0^∞ x^(s-1) e^(-x) dx")
print("-" * 60)
print(f"   Γ(5) = {Gamma(5):.10f}   4! = {math.factorial(4)}   ίσα: {Gamma(5) == math.factorial(4)}")
print(f"   Γ(1/2) = {Gamma(0.5):.12f}   √π = {np.sqrt(np.pi):.12f}")
print(f"   Γ(5/2) = {Gamma(2.5):.12f}   3√π/4 = {3*np.sqrt(np.pi)/4:.12f}")
print(f"   B(2,3) = {Beta(2.0, 3.0):.12f}   Γ(2)Γ(3)/Γ(5) = "
      f"{Gamma(2.0)*Gamma(3.0)/Gamma(5.0):.12f}   (1/12 = {1/12:.12f})")
print("   Ορισμός μέσω ολοκληρώματος (quad):")
for s in (0.5, 2.5, 5.0):
    I, _ = quad(lambda t, s=s: t**(s - 1) * np.exp(-t), 0, np.inf)
    print(f"      s = {s:3.1f} :  ∫ = {I:.10f}   Γ(s) = {Gamma(s):.10f}")

# ── Δ. Προαιρετικό: ολοκληρώματα Β' είδους ∫_0^1 x^(-p) dx ──────────────────
print()
print("Δ. (Προαιρετικό) ∫_0^1 x^(-p) dx  —  Β' είδους, ιδιαίτερο σημείο x=0")
print("-" * 60)
for p in [sp.Rational(1, 2), sp.Integer(1), sp.Integer(2)]:
    I = sp.integrate(x**(-p), (x, 0, 1))
    status = "σύγκλιση" if I.is_finite else "απόκλιση"
    print(f"   p = {str(p):3s} :  ∫ = {str(I):12s}  ({status})")
print("   Συμπέρασμα: συγκλίνει ΑΚΡΙΒΩΣ όταν p < 1.")
print("   (αναμενόμενα: p=1/2 -> 2, p=1 -> oo, p=2 -> oo)")
print()
print("   ΣΥΝΟΨΗ:")
print("     ∫_1^∞ x^(-p) dx  συγκλίνει  <=>  p > 1   (ουρά στο άπειρο)")
print("     ∫_0^1 x^(-p) dx  συγκλίνει  <=>  p < 1   (ιδιομορφία στο 0)")
print("     Καμία τιμή του p δεν κάνει και τα δύο να συγκλίνουν ταυτόχρονα,")
print("     άρα το ∫_0^∞ x^(-p) dx αποκλίνει για κάθε p.")

# ── Γραφήματα ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

xv = np.linspace(1, 10, 500)
axes[0].plot(xv, 1 / xv**2, 'b-', label=r'$1/x^2$ (συγκλίνει)')
axes[0].fill_between(xv, 1 / xv**2, alpha=0.25, color='blue')
axes[0].plot(xv, 1 / xv, 'r-', label=r'$1/x$ (αποκλίνει)')
axes[0].plot(xv, 1 / np.sqrt(xv), 'g--', label=r'$1/\sqrt{x}$ (αποκλίνει)')
axes[0].set_ylim(0, 1.2)
axes[0].set_xlabel('x')
axes[0].legend(fontsize=8)
axes[0].set_title(r"A' είδους: $\int_1^\infty x^{-p}dx$")
axes[0].grid(True, alpha=0.3)

xv2 = np.linspace(0.005, 1, 600)
axes[1].plot(xv2, 1 / np.sqrt(xv2), 'g-', label=r'$1/\sqrt{x}$ (συγκλίνει)')
axes[1].fill_between(xv2, 1 / np.sqrt(xv2), alpha=0.25, color='green')
axes[1].plot(xv2, 1 / xv2, 'r-', label=r'$1/x$ (αποκλίνει)')
axes[1].set_ylim(0, 20)
axes[1].set_xlabel('x')
axes[1].legend(fontsize=8)
axes[1].set_title(r"B' είδους: $\int_0^1 x^{-p}dx$")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()

# Γράφημα της Γ(x) στο (0,5]  (ερώτημα γ)
plt.figure(figsize=(7, 4.5))
xg = np.linspace(0.05, 5.0, 800)
plt.plot(xg, Gamma(xg), 'b-', lw=2, label=r'$\Gamma(x)$')
ints = np.array([1, 2, 3, 4, 5])
plt.plot(ints, Gamma(ints.astype(float)), 'ro',
         label=r'$\Gamma(n)=(n-1)!$')
for m in ints:
    plt.annotate(f'{math.factorial(m - 1)}', (m, Gamma(float(m))),
                 textcoords='offset points', xytext=(6, 4), fontsize=8)
plt.axvline(0, color='gray', lw=0.8)
plt.ylim(0, 26)
plt.xlim(0, 5.2)
plt.xlabel('x')
plt.ylabel(r'$\Gamma(x)$')
plt.title(r'Η συνάρτηση $\Gamma(x)$ στο $(0,5]$ — απειρίζεται καθώς $x\to 0^+$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
