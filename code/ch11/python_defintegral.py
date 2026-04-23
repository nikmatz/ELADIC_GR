# ============================================================
# python_defintegral.py
# Κεφάλαιο 11 — Ορισμένο Ολοκλήρωμα
# Βιβλίο: Μαθηματικά Ι · ΑΣΠΕΤΕ
# Συγγραφέας: Ν. Ματζάκος
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy              → ορισμένο ολοκλήρωμα (ακριβές)
#   scipy.integrate    → quad (αριθμητικό), dblquad
#   numpy              → κανόνας τραπεζίου, Simpson
#   matplotlib         → εμβαδόν, γεωμετρικές παραστάσεις
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.integrate(f, (x, a, b))    → ακριβές ορισμένο
#   scipy.integrate.quad(f, a, b)    → αριθμητικό
#   np.trapz(y, x)                   → κανόνας τραπεζίου
#   scipy.integrate.simpson(y, x)    → κανόνας Simpson
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from sympy import (symbols, integrate, sin, cos, exp, log,
                   sqrt, Abs, pi, E, Rational, simplify,
                   lambdify, solve, oo, pprint)
from scipy.integrate import quad, simpson as sp_simpson
from scipy.integrate import dblquad

x = symbols('x')

print("=" * 55)
print(" Κεφάλαιο 11: Ορισμένο Ολοκλήρωμα — Python")
print("=" * 55)

# ── Α. Βασικά Ορισμένα Ολοκληρώματα ─────────────────────
print("\n── Α. Βασικά Ορισμένα Ολοκληρώματα ──")

def_integrals = [
    (x**2,            0, 3,      "∫₀³ x² dx",         Rational(9,1)),
    (sin(x),          0, pi,     "∫₀^π sin x dx",      2),
    (exp(x),          0, 1,      "∫₀¹ eˣ dx",          E - 1),
    (1/x,             1, E,      "∫₁^e (1/x) dx",      1),
    (x**3-2*x+1,     -1, 2,      "∫₋₁² (x³-2x+1) dx", Rational(15,4)),
    (cos(x),          0, pi/2,   "∫₀^(π/2) cos x dx",  1),
    (1/(x**2+1),      0, 1,      "∫₀¹ 1/(x²+1) dx",    pi/4),
]

print(f"{'Ολοκλήρωμα':<28} {'Ακριβές':>12}  {'float':>12}  {'✓'}")
print("-" * 60)
for f_sym, a, b, label, expected in def_integrals:
    val = integrate(f_sym, (x, a, b))
    val_f = float(val)
    exp_f = float(expected)
    ok = abs(val_f - exp_f) < 1e-9
    print(f"  {label:<26}  {str(val):>12}  {val_f:>12.8f}  {'✓' if ok else '✗'}")

# ── Β. Ιδιότητες ─────────────────────────────────────────
print("\n── Β. Ιδιότητες Ορισμένου Ολοκληρώματος ──")

f_prop = x**2 - x
I_0_2 = float(integrate(f_prop, (x, 0, 2)))
I_0_1 = float(integrate(f_prop, (x, 0, 1)))
I_1_2 = float(integrate(f_prop, (x, 1, 2)))
print(f"  Πρόσθεση: ∫₀²f = ∫₀¹f + ∫₁²f  →  "
      f"{I_0_2:.6f} = {I_0_1:.6f} + {I_1_2:.6f}  "
      f"{'✓' if abs(I_0_2-(I_0_1+I_1_2))<1e-10 else '✗'}")

I_fwd = float(integrate(x**3, (x, 1, 3)))
I_bwd = float(integrate(x**3, (x, 3, 1)))
print(f"  Αντιστροφή: ∫₁³ x³ = {I_fwd:.4f},  ∫₃¹ x³ = {I_bwd:.4f}  "
      f"(άθροισμα={I_fwd+I_bwd:.4f} ✓)")

# ── Γ. Εμβαδόν μεταξύ Καμπυλών ──────────────────────────
print("\n── Γ. Εμβαδόν μεταξύ Καμπυλών ──")

areas = [
    (x,       x**2,   0,  1,    "y=x  vs  y=x²      [0,1]"),
    (sin(x),  cos(x), 0,  pi/4, "sinx vs cosx       [0,π/4]"),
    (-x**2+4, x**2-2*x, -1, 2, "y=-x²+4 vs y=x²-2x [-1,2]"),
]

for f_up, f_dn, a, b, label in areas:
    A = integrate(f_up - f_dn, (x, a, b))
    print(f"  {label}")
    print(f"    Εμβαδόν = {A} = {float(A):.6f}")

# ── Δ. Αριθμητική Ολοκλήρωση ─────────────────────────────
print("\n── Δ. Αριθμητική Ολοκλήρωση (scipy.quad) ──")

numerical_cases = [
    (lambda t: np.exp(t**2),      0, 1,       "∫₀¹ eˣ² dx        (υπερβατικό)"),
    (lambda t: np.sin(t**2),      0, 1,       "∫₀¹ sin(x²) dx    (Fresnel)"),
    (lambda t: np.sin(t)/t if t != 0 else 1.0, 1e-10, np.pi, "∫₀^π sinc(x) dx"),
    (lambda t: 1/np.sqrt(1-t**2+1e-12), 0, 0.99, "∫₀¹ 1/√(1-x²) dx  (→ π/2)"),
]

for f_np, a, b, label in numerical_cases:
    val, err = quad(f_np, a, b)
    print(f"  {label}")
    print(f"    ≈ {val:.10f}  (σφάλμα ≤ {err:.2e})")

# ── Ε. Κανόνας Τραπεζίου & Simpson ───────────────────────
print("\n── Ε. Κανόνας Τραπεζίου & Simpson ──")

def trapezoid_rule(f, a, b, n):
    xi = np.linspace(a, b, n+1)
    yi = f(xi)
    return np.trapz(yi, xi)

def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        n += 1
    xi = np.linspace(a, b, n+1)
    yi = f(xi)
    return sp_simpson(yi, x=xi)

# Σύγκριση για ∫₀¹ x² dx (ακριβές = 1/3)
f_test = lambda t: t**2
exact  = 1/3
print(f"  ∫₀¹ x² dx  (ακριβές = {exact:.10f})")
for n in [4, 8, 16, 32]:
    T = trapezoid_rule(f_test, 0, 1, n)
    S = simpsons_rule(f_test, 0, 1, n)
    print(f"    n={n:2d}:  Τραπέζιο={T:.10f} (σφ={abs(T-exact):.2e})"
          f"   Simpson={S:.10f} (σφ={abs(S-exact):.2e})")

# Σύγκριση για ∫₀^π sinx dx (ακριβές = 2)
f_sin = np.sin
exact2 = 2.0
print(f"\n  ∫₀^π sinx dx  (ακριβές = 2)")
for n in [4, 8, 16]:
    T = trapezoid_rule(f_sin, 0, np.pi, n)
    S = simpsons_rule(f_sin, 0, np.pi, n)
    print(f"    n={n:2d}:  Τραπέζιο={T:.8f} (σφ={abs(T-exact2):.2e})"
          f"   Simpson={S:.8f} (σφ={abs(S-exact2):.2e})")

# ── Στ. Γραφικές Παραστάσεις ────────────────────────────
print("\n── Στ. Γραφικές Παραστάσεις ──")

fig = plt.figure(figsize=(14, 9))
fig.suptitle("Ορισμένο Ολοκλήρωμα", fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, fig, hspace=0.45, wspace=0.35)

# 1. ∫₀³ x² dx — σκιασμένη περιοχή
ax1 = fig.add_subplot(gs[0, 0])
t1 = np.linspace(-0.3, 3.5, 400)
y1 = t1**2
ax1.plot(t1, y1, 'royalblue', lw=2.5, label=r'$f(x)=x^2$')
tx = np.linspace(0, 3, 200)
ax1.fill_between(tx, tx**2, alpha=0.25, color='royalblue',
                  label=r'$\int_0^3 x^2\,dx=9$')
ax1.axhline(0, color='k', lw=0.5)
ax1.set_xlim(-0.3, 3.5); ax1.set_ylim(-0.5, 10)
ax1.set_title(r"$\int_0^3 x^2\,dx = 9$", fontsize=10)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# 2. Εμβαδόν μεταξύ y=x και y=x²
ax2 = fig.add_subplot(gs[0, 1])
t2 = np.linspace(-0.2, 1.3, 400)
ax2.plot(t2, t2,    'royalblue', lw=2.5, label=r'$y=x$')
ax2.plot(t2, t2**2, 'tomato',    lw=2.5, label=r'$y=x^2$')
tx2 = np.linspace(0, 1, 200)
ax2.fill_between(tx2, tx2**2, tx2, alpha=0.25, color='green',
                  label=r'$A=\frac{1}{6}$')
ax2.axhline(0, color='k', lw=0.5)
ax2.set_title(r"Εμβαδόν: $y=x$ vs $y=x^2$", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# 3. ∫₀^π sinx dx = 2
ax3 = fig.add_subplot(gs[0, 2])
t3 = np.linspace(-0.2, np.pi+0.2, 400)
ax3.plot(t3, np.sin(t3), 'royalblue', lw=2.5, label=r'$\sin x$')
tx3 = np.linspace(0, np.pi, 200)
ax3.fill_between(tx3, np.sin(tx3), alpha=0.25, color='royalblue',
                  label=r'$\int_0^\pi \sin x\,dx=2$')
ax3.axhline(0, color='k', lw=0.8)
ax3.set_xticks([0, np.pi/2, np.pi])
ax3.set_xticklabels(['0', 'π/2', 'π'])
ax3.set_title(r"$\int_0^\pi \sin x\,dx = 2$", fontsize=10)
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# 4. Εμβαδόν μεταξύ παραβολών
ax4 = fig.add_subplot(gs[1, 0])
t4 = np.linspace(-1.5, 2.5, 400)
y4a = -t4**2 + 4
y4b = t4**2 - 2*t4
ax4.plot(t4, y4a, 'royalblue', lw=2.5, label=r'$y=-x^2+4$')
ax4.plot(t4, y4b, 'tomato',    lw=2.5, label=r'$y=x^2-2x$')
tx4 = np.linspace(-1, 2, 200)
ax4.fill_between(tx4, tx4**2-2*tx4, -tx4**2+4, alpha=0.25, color='green',
                  label='Εμβαδόν=9')
ax4.axhline(0, color='k', lw=0.5)
ax4.set_ylim(-3, 6)
ax4.set_title("Εμβαδόν μεταξύ παραβολών", fontsize=10)
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

# 5. Σύγκριση Τραπεζίου vs Simpson (σφάλμα)
ax5 = fig.add_subplot(gs[1, 1])
ns   = [2, 4, 8, 16, 32, 64]
trap_errs = [abs(trapezoid_rule(lambda t: t**2, 0, 1, n) - 1/3) for n in ns]
simp_errs = [abs(simpsons_rule( lambda t: t**2, 0, 1, n) - 1/3) for n in ns]
ax5.loglog(ns, trap_errs, 'o-', color='royalblue', lw=2, label='Τραπέζιο')
ax5.loglog(ns, simp_errs, 's-', color='tomato',    lw=2, label='Simpson')
ax5.set_xlabel('n'); ax5.set_ylabel('|σφάλμα|')
ax5.set_title(r"Σύγκλιση: $\int_0^1 x^2\,dx$", fontsize=10)
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3, which='both')

# 6. Αριθμητική vs Ακριβής (bar chart σφαλμάτων)
ax6 = fig.add_subplot(gs[1, 2])
labels6 = [r'$\int_0^1 x^2$', r'$\int_0^\pi \sin x$', r'$\int_0^1 e^x$', r'$\int_1^e 1/x$']
exact6  = [1/3, 2, np.e-1, 1]
num6    = [quad(lambda t: t**2, 0, 1)[0],
           quad(np.sin, 0, np.pi)[0],
           quad(np.exp, 0, 1)[0],
           quad(lambda t: 1/t, 1, np.e)[0]]
errs6   = [abs(n-e) for n,e in zip(num6, exact6)]
bars = ax6.bar(labels6, errs6, color=['royalblue','tomato','forestgreen','orange'])
ax6.set_ylabel('|σφάλμα|'); ax6.set_yscale('log')
ax6.set_title("scipy.quad: Σφάλμα", fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')
for bar, err in zip(bars, errs6):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.5,
             f'{err:.0e}', ha='center', va='bottom', fontsize=7)

plt.savefig("definite_integrals.png", dpi=120, bbox_inches='tight')
print("Το διάγραμμα αποθηκεύτηκε: definite_integrals.png")
plt.show()
