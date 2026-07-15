# python_seires.py — Ακολουθίες και Σειρές (Κεφ. 19)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# Απαιτεί: pip install sympy matplotlib numpy

from sympy import *
import numpy as np
import matplotlib.pyplot as plt

n, x = symbols('n x', real=True)
k = symbols('k', integer=True, positive=True)

print("=" * 60)
print("ΑΚΟΛΟΥΘΙΕΣ ΚΑΙ ΣΕΙΡΕΣ")
print("=" * 60)

# ── 1. Όριο ακολουθίας ───────────────────────────────────────────────────────
print()
print("1. ΟΡΙΑ ΑΚΟΛΟΥΘΙΩΝ")
seqs = [
    (n / (n + 1), "n/(n+1)"),
    ((1 + 1/n)**n, "(1+1/n)^n → e"),
    (n**2 / exp(n), "n²/eⁿ"),
    (sin(n)/n, "sin(n)/n"),
]
for expr, label in seqs:
    lim = limit(expr, n, oo)
    print(f"   lim {label} = {lim}")

# ── 2. Γεωμετρική Σειρά ──────────────────────────────────────────────────────
print()
print("2. ΓΕΩΜΕΤΡΙΚΗ ΣΕΙΡΑ  Σ qⁿ")
q = symbols('q')
for qval in [Rational(1,2), Rational(1,3), Rational(2,3), -Rational(1,2)]:
    S = summation(qval**k, (k, 0, oo))
    print(f"   q = {qval}: Σ qⁿ = {S}")

# ── 3. Σειρά Taylor / Maclaurin ─────────────────────────────────────────────
print()
print("3. ΑΝΑΠΤΥΓΜΑ TAYLOR / MACLAURIN")
funcs = [
    (exp(x), "e^x", 0, 6),
    (sin(x), "sin(x)", 0, 7),
    (cos(x), "cos(x)", 0, 7),
    (1/(1-x), "1/(1-x)", 0, 5),
    (ln(1+x), "ln(1+x)", 0, 6),
]
for f, label, a0, order in funcs:
    T = series(f, x, a0, order)
    print(f"   {label} = {T}")

# ── 4. Κριτήρια Σύγκλισης ───────────────────────────────────────────────────
print()
print("4. ΚΡΙΤΗΡΙΟ ΛΟΓΟΥ (ratio test) — an = n!/nⁿ")
a_n = factorial(n) / n**n
ratio = simplify(a_n.subs(n, n+1) / a_n)
L = limit(ratio, n, oo)
print(f"   |a_{{n+1}}/a_n| → {L}  (< 1 → σύγκλιση)")

# ── 5. Ακτίνα σύγκλισης ─────────────────────────────────────────────────────
print()
print("5. ΑΚΤΙΝΑ ΣΥΓΚΛΙΣΗΣ — Σ xⁿ/n²")
c_n = x**n / n**2
c_ratio = simplify(abs(c_n.subs(n, n+1) / c_n))
R = 1 / limit(c_ratio / abs(x), n, oo)
print(f"   Ακτίνα R = {R}")

# ── 6. Γράφημα μερικών αθροισμάτων ─────────────────────────────────────────
print()
xv = np.linspace(-1, 1, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Maclaurin του sin(x)
true_sin = np.sin(xv)
axes[0].plot(xv, true_sin, 'k-', linewidth=2, label='sin(x)')
for order in [1, 3, 5, 7]:
    taylor_val = sum(
        ((-1)**m * xv**(2*m+1)) / float(factorial(2*m+1))
        for m in range((order+1)//2)
    )
    axes[0].plot(xv, taylor_val, '--', label=f'Τάξη {order}')
axes[0].set_ylim(-1.5, 1.5)
axes[0].legend(fontsize=8); axes[0].set_title('Taylor του sin(x)')
axes[0].grid(True, alpha=0.3)

# Γεωμετρική σειρά q=1/2
partial_sums = [sum(0.5**i for i in range(n_terms)) for n_terms in range(1, 15)]
ns = list(range(1, 15))
axes[1].plot(ns, partial_sums, 'bo-', label='Μερικά αθρ. Σ(1/2)ⁿ')
axes[1].axhline(2, color='r', linestyle='--', label='Όριο = 2')
axes[1].legend(); axes[1].set_title('Γεωμετρική σειρά, q=1/2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig('seires.png', dpi=100); plt.show()
print("Αρχείο 'seires.png' αποθηκεύτηκε.")
