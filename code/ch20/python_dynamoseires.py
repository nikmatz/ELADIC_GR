# python_dynamoseires.py — Δυναμοσειρές, Σύγκλιση, Γραφικά (Κεφ. 20)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# Εντολές: sympy.series(), sympy.diff(), sympy.integrate(), math.factorial(),
#          scipy.integrate.quad(), matplotlib.pyplot

import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

x = sp.symbols('x', real=True)
n = sp.symbols('n', positive=True, integer=True)

print("=" * 66)
print("ΔΥΝΑΜΟΣΕΙΡΕΣ: ΑΚΤΙΝΑ ΣΥΓΚΛΙΣΗΣ, MACLAURIN, π, ΟΛΟΚΛΗΡΩΜΑΤΑ")
print("=" * 66)

# ── Α. Πίνακας ακτίνων σύγκλισης + αναπτύγματα Maclaurin ────────────────────
print()
print("Α1. ΠΙΝΑΚΑΣ ΑΚΤΙΝΩΝ ΣΥΓΚΛΙΣΗΣ   R = lim |c_n / c_(n+1)|")
print("-" * 66)
# Υπολογίζουμε κατευθείαν το lim |c_n/c_(n+1)|: έτσι το R = ∞ προκύπτει
# φυσιολογικά, χωρίς διαίρεση με το 0.
coeffs = [
    (1 / n,               "Σ xⁿ/n",   "1"),
    (1 / sp.factorial(n), "Σ xⁿ/n!",  "oo"),
    (n,                   "Σ n·xⁿ",   "1"),
    (1 / 2**n,            "Σ xⁿ/2ⁿ",  "2"),
    (sp.factorial(n),     "Σ n!·xⁿ",  "0"),
]
print(f"   {'σειρά':12s} {'R':>6s}   αναμενόμενο")
for c, label, expected in coeffs:
    R = sp.limit(sp.Abs(c / c.subs(n, n + 1)), n, sp.oo)
    print(f"   {label:12s} {str(R):>6s}   ({expected})")
print("   R = oo: συγκλίνει για κάθε x.   R = 0: μόνο στο x = 0.")

print()
print("Α2. ΑΝΑΠΤΥΓΜΑΤΑ MACLAURIN  sympy.series(f, x, 0, 8)")
print("-" * 66)
for f in [sp.exp(x), sp.sin(x), sp.cos(x), sp.log(1 + x),
          1 / (1 - x), sp.atan(x), sp.sqrt(1 + x)]:
    print(f"   {str(f):10s} = {sp.series(f, x, 0, 8)}")
print(f"   Taylor του e^x γύρω από x0=1: {sp.series(sp.exp(x), x, 1, 5)}")

# ── Β. Γραφικά Maclaurin για τάξεις 1, 3, 5, 9 ──────────────────────────────
print()
print("Β. ΠΟΛΥΩΝΥΜΑ MACLAURIN — ΠΟΥ ΑΠΟΚΛΙΝΕΙ ΤΟ ΚΑΘΕΝΑ")
print("-" * 66)
cases = [
    (sp.exp(x),      'e^x',        (-3.0,  3.0),  (-2, 12),   'R = oo'),
    (sp.sin(x),      'sin x',      (-7.0,  7.0),  (-2.5, 2.5), 'R = oo'),
    (sp.log(1 + x),  'ln(1+x)',    (-0.95, 2.0),  (-4, 2),    'R = 1'),
    (sp.atan(x),     'arctan x',   (-2.0,  2.0),  (-3, 3),    'R = 1'),
]
orders = (1, 3, 5, 9)
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for ax, (f, name, (xa, xb), ylim, Rtxt) in zip(axes.ravel(), cases):
    xv = np.linspace(xa, xb, 600)
    fnum = sp.lambdify(x, f, 'numpy')
    ax.plot(xv, fnum(xv), 'k-', lw=2.2, label=name)
    for o in orders:
        P = sp.series(f, x, 0, o + 1).removeO()
        Pnum = sp.lambdify(x, P + 0 * x, 'numpy')
        ax.plot(xv, Pnum(xv), '--', lw=1.2, label=f'τάξη {o}')
    if Rtxt == 'R = 1':
        ax.axvline(-1, color='gray', ls=':', lw=1)
        ax.axvline(1, color='gray', ls=':', lw=1)
    ax.set_ylim(*ylim)
    ax.set_title(f'{name}   ({Rtxt})')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
plt.tight_layout()

# Αριθμητική τεκμηρίωση: πόσο απέχει το πολυώνυμο τάξης 9 σε διάφορα x
for f, name, _, _, Rtxt in cases:
    P9 = sp.lambdify(x, sp.series(f, x, 0, 10).removeO() + 0 * x, 'numpy')
    fn = sp.lambdify(x, f, 'numpy')
    pts = [0.5, 0.9, 1.1, 2.0] if Rtxt == 'R = 1' else [0.5, 1.0, 3.0, 6.0]
    errs = ", ".join(f"x={p}: {abs(float(P9(p)) - float(fn(p))):.2e}" for p in pts)
    print(f"   |P9 - {name:9s}|  {errs}   ({Rtxt})")
print("   Για R = 1 το σφάλμα εκρήγνυται μόλις περάσουμε το |x| = 1·")
print("   για R = oo φθίνει παντού, απλώς πιο αργά όσο απομακρυνόμαστε.")

# ── Γ. Υπολογισμός του π ────────────────────────────────────────────────────
print()
print("Γ. ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ π: Gregory–Leibniz (αργή) vs Machin (ταχεία)")
print("-" * 66)

def atan_series(z, K):
    """Σ_{k=0}^{K-1} (-1)^k z^(2k+1)/(2k+1) — η σειρά του arctan."""
    return sum((-1)**k * z**(2 * k + 1) / (2 * k + 1) for k in range(K))

def gregory_leibniz(N):
    """π/4 = Σ_{k=0}^{N} (-1)^k/(2k+1) — η σειρά του arctan για z = 1."""
    return 4.0 * atan_series(1.0, N + 1)

def machin(K):
    return 4.0 * (4 * atan_series(1 / 5, K) - atan_series(1 / 239, K))

for N in (10, 100, 1000):
    v = gregory_leibniz(N)
    print(f"   Gregory–Leibniz, N = {N:5d}: {v:.12f}   σφάλμα {abs(v - np.pi):.3e}")
for K in (2, 5, 10):
    v = machin(K)
    print(f"   Machin,          {K:5d} όροι: {v:.12f}   σφάλμα {abs(v - np.pi):.3e}")
print(f"   π = {np.pi:.12f}")
print("   Η Gregory–Leibniz κερδίζει ~1 ψηφίο ανά 10πλασιασμό των όρων (1/N)·")
print("   ο Machin κερδίζει ~1.4 ψηφία ΑΝΑ ΟΡΟ (γεωμετρική σύγκλιση, λόγος 1/25).")

Ks = np.arange(1, 61)
gl_err = np.array([abs(gregory_leibniz(int(K) - 1) - np.pi) for K in Ks])
ma_err = np.array([max(abs(machin(int(K)) - np.pi), 1e-17) for K in Ks])
plt.figure(figsize=(7.5, 4.5))
plt.semilogy(Ks, gl_err, 'b.-', ms=4, label='Gregory–Leibniz')
plt.semilogy(Ks, ma_err, 'r.-', ms=4, label='Τύπος Machin')
plt.xlabel('πλήθος όρων K'); plt.ylabel('|σφάλμα|')
plt.title('Σύγκλιση προς το π (λογαριθμικός άξονας σφάλματος)')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()

# ── Δ. ∫_0^1 sin(x²) dx μέσω δυναμοσειράς — σφάλμα ανά τάξη ─────────────────
print()
print("Δ. ∫_0^1 sin(x²) dx  —  δυναμοσειρά έναντι scipy.quad")
print("-" * 66)
# sin(x²) = Σ (-1)^k x^(4k+2)/(2k+1)!  ->  ∫_0^1 = Σ (-1)^k / ((4k+3)(2k+1)!)
qv, qerr = quad(lambda t: np.sin(t**2), 0, 1)
print(f"   scipy.quad = {qv:.12f}   (εκτ. σφάλμα {qerr:.1e})")
print()
print(f"   {'όροι K':>7s} {'τάξη x^(4K-2)':>14s} {'προσέγγιση':>16s} {'|σφάλμα|':>12s}")
partial = 0.0
for K in range(1, 8):
    k = K - 1
    partial += (-1)**k / ((4 * k + 3) * math.factorial(2 * k + 1))
    print(f"   {K:7d} {4*K - 2:14d} {partial:16.12f} {abs(partial - qv):12.2e}")
print("   Κάθε επιπλέον όρος κερδίζει περίπου δύο δεκαδικά ψηφία: με 5 όρους")
print("   είμαστε στο ~1e-9 και με 7 στα όρια της διπλής ακρίβειας.")

# Σύγκριση με τη συμβολική τιμή του SymPy
sym = sp.integrate(sp.sin(x**2), (x, 0, 1))
print(f"   SymPy συμβολικά: {sym}")
print(f"   αριθμητικά     : {float(sym.evalf()):.12f}   (0.310268301723)")
plt.show()
