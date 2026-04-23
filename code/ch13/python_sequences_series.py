# ============================================================
# python_sequences_series.py
# Κεφάλαιο 13 — Ακολουθίες και Σειρές
# Βιβλίο: Μαθηματικά Ι · ΑΣΠΕΤΕ
# Συγγραφέας: Ν. Ματζάκος
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy              → limits, symbolic sum, Taylor
#   numpy              → αριθμητικές ακολουθίες & σειρές
#   matplotlib         → γραφική σύγκλιση, μερικά αθροίσματα
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.limit(a_n, n, oo)           → όριο ακολουθίας
#   sympy.summation(a_n, (n, 1, oo))  → άθροισμα σειράς
#   sympy.series(f, x, 0, n)          → σειρά Taylor/Maclaurin
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import (symbols, limit, oo, summation, Sum, factorial,
                   series, sin, cos, exp, log, Rational, pi,
                   sqrt, simplify, lambdify, pprint, S, zoo)
from math import factorial as mfact

n, k, x = symbols('n k x', positive=True)

print("=" * 55)
print(" Κεφάλαιο 13: Ακολουθίες και Σειρές — Python")
print("=" * 55)

# ── Α. Ακολουθίες — Σύγκλιση / Απόκλιση ──────────────────
print("\n── Α. Ακολουθίες ──")

sequences = [
    (n/(n+1),           "n/(n+1)",      True),
    ((2*n**2-1)/n**2,   "(2n²-1)/n²",  True),
    ((-1)**n/n,         "(-1)ⁿ/n",     True),
    ((1 + 1/n)**n,      "(1+1/n)ⁿ",   True),
    (n**2/exp(n),       "n²/eⁿ",       True),
    (n,                 "n",           False),
]

print(f"{'ακολουθία':<18} {'lim':>12}  {'σύγκλιση'}")
print("-" * 44)
for a_n, label, conv in sequences:
    try:
        L = limit(a_n, n, oo)
        converges = (L != oo and L != -oo and L != zoo)
        print(f"  {label:<16}  {str(L):>14}  {'✓ Συγκλίνει' if converges else '✗ Αποκλίνει'}")
    except Exception as e:
        print(f"  {label:<16}  {'?':>14}  (βλ. sympy)")

# Πρώτοι 10 όροι αριθμητικά
print("\nΠρώτοι 10 όροι aₙ = (1+1/n)ⁿ (→ e):")
terms_e = [(1 + 1/i)**i for i in range(1, 11)]
for i, v in enumerate(terms_e, 1):
    print(f"  a_{i:2d} = {v:.8f}  (σφάλμα vs e: {abs(v - np.e):.2e})")

# ── Β. Γεωμετρικές Σειρές ─────────────────────────────────
print("\n── Β. Γεωμετρικές Σειρές ──")

geo_cases = [
    (Rational(1,2),  "1/2"),
    (Rational(2,3),  "2/3"),
    (Rational(-1,3), "-1/3"),
    (Rational(1,4),  "1/4"),
]

print(f"{'r':>6}  {'Σ rⁿ (n=0..∞)':>16}  {'float':>12}")
for r, rlabel in geo_cases:
    S = summation(r**n, (n, 0, oo))
    print(f"  r={rlabel:<6}  {str(S):>16}  {float(S):>12.8f}")

# Τηλεσκοπική Σ 1/(n(n+1))
print("\nΤηλεσκοπική Σ 1/(n(n+1)) = 1:")
S_tel = summation(1/(n*(n+1)), (n, 1, oo))
print(f"  Ακριβές: {S_tel}")
print("  Μερικά αθροίσματα S_N:")
partial = [sum(1/(i*(i+1)) for i in range(1, N+1)) for N in [5, 10, 20, 50]]
for N, SN in zip([5, 10, 20, 50], partial):
    print(f"    N={N:3d}: S_N = {SN:.10f}  (σφάλμα={abs(SN-1):.2e})")

# ── Γ. Κριτήρια Σύγκλισης ─────────────────────────────────
print("\n── Γ. Κριτήρια Σύγκλισης ──")

# Κριτήριο λόγου (D'Alembert) αριθμητικά
def ratio_test(a_func, N=50):
    """Υπολογίζει |a_{n+1}/a_n| για μεγάλο n"""
    ratios = [abs(a_func(i+1)/a_func(i)) for i in range(1, N+1)]
    return ratios[-1]

ratio_cases = [
    (lambda i: mfact(i)/i**i,   "Σ n!/nⁿ",   "< 1 → ΣΥΓΚΛΙΝΕΙ"),
    (lambda i: i**i/mfact(i),   "Σ nⁿ/n!",   "> 1 → ΑΠΟΚΛΙΝΕΙ"),
    (lambda i: i**2/2**i,       "Σ n²/2ⁿ",   "< 1 → ΣΥΓΚΛΙΝΕΙ"),
    (lambda i: 2**i/mfact(i),   "Σ 2ⁿ/n!",   "< 1 → ΣΥΓΚΛΙΝΕΙ"),
]

print("Κριτήριο Λόγου (|a_{n+1}/a_n| για n=50):")
for a_func, label, verdict in ratio_cases:
    try:
        L = ratio_test(a_func)
        print(f"  {label:<14}: L ≈ {L:.6f}  {verdict}")
    except OverflowError:
        print(f"  {label:<14}: overflow — {verdict}")

# Σειρά Basel: Σ 1/n² = π²/6
print("\n── Γ2. Σύγκλιση Σ 1/n² → π²/6 ──")
exact = np.pi**2/6
for N in [10, 100, 1_000, 10_000]:
    SN = sum(1/i**2 for i in range(1, N+1))
    print(f"  N={N:6d}: S_N = {SN:.10f}  σφάλμα = {abs(SN-exact):.2e}")

# ── Δ. Σειρές Taylor / Maclaurin ─────────────────────────
print("\n── Δ. Σειρές Maclaurin ──")

maclaurin_cases = [
    (exp(x),    "eˣ"),
    (sin(x),    "sin x"),
    (cos(x),    "cos x"),
    (1/(1-x),   "1/(1-x)"),
    (log(1+x),  "ln(1+x)"),
]

for f_sym, label in maclaurin_cases:
    T = series(f_sym, x, 0, 7)
    print(f"  {label}: {T}")

# Αριθμητική σύγκλιση eˣ στο x=1
print("\nΣύγκλιση Maclaurin eˣ στο x=1:")
for n_terms in [3, 5, 8, 12, 15]:
    approx = sum(1/mfact(k) for k in range(n_terms+1))
    err = abs(approx - np.e)
    print(f"  n={n_terms:2d}: {approx:.12f}  σφάλμα={err:.2e}")

# ── Ε. Γραφικές Παραστάσεις ──────────────────────────────
print("\n── Ε. Γραφικές Παραστάσεις ──")

fig = plt.figure(figsize=(14, 9))
fig.suptitle("Ακολουθίες και Σειρές", fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, fig, hspace=0.45, wspace=0.35)

# 1. Ακολουθία (1+1/n)ⁿ → e
ax1 = fig.add_subplot(gs[0, 0])
ns1  = np.arange(1, 51)
an1  = (1 + 1/ns1)**ns1
ax1.plot(ns1, an1, 'o-', color='royalblue', ms=4, lw=1.5, label=r'$(1+1/n)^n$')
ax1.axhline(np.e, color='tomato', ls='--', lw=2, label=f'$e={np.e:.4f}$')
ax1.set_xlabel('n'); ax1.set_title(r"$(1+\frac{1}{n})^n \to e$", fontsize=10)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# 2. Γεωμετρική σειρά: μερικά αθροίσματα Σ (1/2)ⁿ
ax2 = fig.add_subplot(gs[0, 1])
ns2  = np.arange(0, 20)
an2  = 0.5**ns2
SN2  = np.cumsum(an2)
ax2.bar(ns2, an2, alpha=0.5, color='royalblue', label=r'$a_n=(1/2)^n$')
ax2.plot(ns2, SN2, 'o-', color='tomato', ms=5, lw=1.8, label=r'$S_N \to 2$')
ax2.axhline(2, color='tomato', ls='--', lw=1.5)
ax2.set_xlabel('N'); ax2.set_title(r"$\sum(1/2)^n \to 2$", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# 3. Σύγκλιση Σ 1/n² → π²/6
ax3 = fig.add_subplot(gs[0, 2])
ns3  = np.arange(1, 200)
SN3  = np.cumsum(1/ns3**2)
ax3.plot(ns3, SN3, 'royalblue', lw=2, label=r'$S_N = \sum_{k=1}^N 1/k^2$')
ax3.axhline(np.pi**2/6, color='tomato', ls='--', lw=2,
            label=r'$\pi^2/6$')
ax3.set_xlabel('N'); ax3.set_title(r"$\sum 1/n^2 \to \pi^2/6$", fontsize=10)
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# 4. Maclaurin για eˣ — σύγκλιση
ax4 = fig.add_subplot(gs[1, 0])
xv4 = np.linspace(-3, 3, 400)
ax4.plot(xv4, np.exp(xv4), 'k', lw=2.5, label=r'$e^x$')
colors4 = ['royalblue', 'tomato', 'forestgreen', 'orange']
for order, col in zip([1, 3, 5, 9], colors4):
    T4 = sum(xv4**k / mfact(k) for k in range(order+1))
    ax4.plot(xv4, T4, ls='--', lw=1.5, color=col,
             label=f'n={order}')
ax4.set_ylim(-5, 15); ax4.set_xlim(-3, 3)
ax4.set_title(r"Maclaurin $e^x$", fontsize=10)
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

# 5. Maclaurin για sinx
ax5 = fig.add_subplot(gs[1, 1])
xv5 = np.linspace(-2*np.pi, 2*np.pi, 500)
ax5.plot(xv5, np.sin(xv5), 'k', lw=2.5, label=r'$\sin x$')
for order, col in zip([1, 3, 5, 9], colors4):
    T5 = sum((-1)**k * xv5**(2*k+1) / mfact(2*k+1) for k in range((order+1)//2 + 1))
    ax5.plot(xv5, T5, ls='--', lw=1.5, color=col, label=f'n={2*((order+1)//2+1)-1}')
ax5.set_ylim(-3, 3); ax5.set_xlim(-2*np.pi, 2*np.pi)
ax5.set_title(r"Maclaurin $\sin x$", fontsize=10)
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# 6. Σφάλμα σύγκλισης Σ 1/n² (log-log)
ax6 = fig.add_subplot(gs[1, 2])
ns6  = np.logspace(0, 4, 100).astype(int)
ns6  = np.unique(ns6)
errs6 = [abs(sum(1/i**2 for i in range(1, N+1)) - np.pi**2/6) for N in ns6]
ax6.loglog(ns6, errs6, 'royalblue', lw=2, label='σφάλμα')
ax6.loglog(ns6, 1/ns6.astype(float), 'tomato', ls='--', lw=1.5, label='1/N')
ax6.set_xlabel('N'); ax6.set_ylabel('|σφάλμα|')
ax6.set_title(r"Σύγκλιση $\sum 1/n^2$ (log-log)", fontsize=10)
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3, which='both')

plt.savefig("sequences_series.png", dpi=120, bbox_inches='tight')
print("Το διάγραμμα αποθηκεύτηκε: sequences_series.png")
plt.show()
