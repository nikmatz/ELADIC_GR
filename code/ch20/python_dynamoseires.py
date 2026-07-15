# python_dynamoseires.py — Δυναμοσειρές και Σειρές Taylor (Κεφ. 20)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# Απαιτεί: pip install sympy numpy matplotlib scipy

from sympy import (symbols, limit, series, diff, integrate, factorial,
                   exp, sin, cos, ln, atan, sqrt, oo, Abs, Rational, simplify)
import numpy as np
import matplotlib.pyplot as plt

x = symbols('x', real=True)
n = symbols('n', integer=True, positive=True)

print("=" * 60)
print("ΔΥΝΑΜΟΣΕΙΡΕΣ ΚΑΙ ΣΕΙΡΕΣ TAYLOR")
print("=" * 60)

# ── 1. Ακτίνα σύγκλισης  R = lim |c_n / c_{n+1}| ─────────────────────────────
print("\n1. ΑΚΤΙΝΑ ΣΥΓΚΛΙΣΗΣ  R = lim |c_n / c_(n+1)|")
coeffs = [
    (Rational(1, 1) / n, "Σ xⁿ/n"),
    (1 / factorial(n), "Σ xⁿ/n!"),
    (n, "Σ n·xⁿ"),
    (1 / 2**n, "Σ xⁿ/2ⁿ"),
    (factorial(n), "Σ n!·xⁿ"),
]
for c_n, label in coeffs:
    R = 1 / limit(Abs(c_n.subs(n, n + 1) / c_n), n, oo)
    Rtxt = "∞" if R in (oo, -oo) or R.is_infinite else str(R)
    print(f"   {label:12s}:  R = {Rtxt}")

# ── 2. Αναπτύγματα Maclaurin (τάξη 8) ───────────────────────────────────────
print("\n2. ΑΝΑΠΤΥΓΜΑ MACLAURIN  (sympy.series, τάξη 8)")
funcs = [exp(x), sin(x), cos(x), ln(1 + x), 1 / (1 - x), atan(x), sqrt(1 + x)]
for f in funcs:
    print(f"   {f} = {series(f, x, 0, 8)}")
print(f"   Taylor e^x γύρω από x0=1: {series(exp(x), x, 1, 4)}")

# ── 3. Πράξεις: παράγωγος & ολοκλήρωμα δυναμοσειράς ──────────────────────────
print("\n3. ΠΡΑΞΕΙΣ ΜΕ ΔΥΝΑΜΟΣΕΙΡΕΣ")
print(f"   d/dx [1/(1-x)] = {simplify(diff(1/(1-x), x))}   (= Σ n·x^(n-1))")
print(f"   ∫ 1/(1+x) dx   = {integrate(1/(1+x), x)}   (= Σ (-1)^n x^(n+1)/(n+1))")

# ── 4. Υπολογισμός του π ─────────────────────────────────────────────────────
print("\n4. ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ π")
def gregory_leibniz(N):
    return 4.0 * sum((-1)**k / (2 * k + 1) for k in range(N + 1))
def machin():
    return 4.0 * (4 * np.arctan(1 / 5) - np.arctan(1 / 239))
for N in (10, 100, 1000):
    val = gregory_leibniz(N)
    print(f"   Gregory-Leibniz N={N:<4d}: {val:.10f}  (σφάλμα {abs(val-np.pi):.2e})")
print(f"   Τύπος Machin        : {machin():.10f}  (σφάλμα {abs(machin()-np.pi):.2e})")

# ── 5. Ολοκλήρωμα μέσω δυναμοσειράς: ∫_0^1 sin(x²) dx ────────────────────────
print("\n5. ∫_0^1 sin(x²) dx ΜΕΣΩ ΔΥΝΑΜΟΣΕΙΡΑΣ")
# sin(x²) = Σ (-1)^k x^(4k+2)/(2k+1)!  →  όρος-όρος: 1/((4k+3)(2k+1)!)
series_val = sum((-1)**k / ((4 * k + 3) * factorial(2 * k + 1)) for k in range(6))
print(f"   Σειρά (6 όροι) = {float(series_val):.10f}")
try:
    from scipy.integrate import quad
    q, _ = quad(lambda t: np.sin(t**2), 0, 1)
    print(f"   scipy.quad     = {q:.10f}")
except Exception as e:
    print(f"   (scipy μη διαθέσιμο: {e})")

# ── 6. Γραφικά: Maclaurin σε αυξανόμενες τάξεις ─────────────────────────────
xv = np.linspace(-2.5, 2.5, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# (α) e^x
axes[0].plot(xv, np.exp(xv), 'k-', lw=2, label='e^x')
for order in (1, 3, 5, 9):
    approx = sum(xv**m / float(factorial(m)) for m in range(order + 1))
    axes[0].plot(xv, approx, '--', label=f'Τάξη {order}')
axes[0].set_ylim(-2, 12); axes[0].legend(fontsize=8)
axes[0].set_title('Maclaurin του e^x'); axes[0].grid(True, alpha=0.3)

# (β) σύγκλιση π (log-scale)
Ns = np.arange(1, 400)
gl_err = [abs(gregory_leibniz(int(N)) - np.pi) for N in Ns]
axes[1].semilogy(Ns, gl_err, 'b-', label='Gregory-Leibniz')
axes[1].axhline(abs(machin() - np.pi) + 1e-16, color='r', ls='--', label='Machin')
axes[1].set_xlabel('N όρων'); axes[1].set_ylabel('|σφάλμα|')
axes[1].legend(); axes[1].set_title('Σύγκλιση προς π'); axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig('dynamoseires.png', dpi=100); plt.show()
print("\nΑρχείο 'dynamoseires.png' αποθηκεύτηκε.")
