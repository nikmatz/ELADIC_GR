# python_seires.py — Ακολουθίες, Σειρές και Κριτήρια Σύγκλισης (Κεφ. 19)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# Εντολές: sympy.limit(), sympy.summation(), np.cumsum(), math.factorial(), matplotlib.pyplot

import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

n = sp.symbols('n', positive=True)
k = sp.symbols('k', integer=True, positive=True)

print("=" * 60)
print("ΑΚΟΛΟΥΘΙΕΣ, ΣΕΙΡΕΣ ΚΑΙ ΚΡΙΤΗΡΙΑ ΣΥΓΚΛΙΣΗΣ")
print("=" * 60)

# ── Α. Όρια ακολουθιών ──────────────────────────────────────────────────────
print()
print("Α. ΟΡΙΑ ΑΚΟΛΟΥΘΙΩΝ  (sympy.limit)")
print("-" * 60)
seqs = [
    (n / (n + 1),        "n/(n+1)",     "1"),
    ((1 + 1/n)**n,       "(1+1/n)^n",   "e"),
    (n**2 / sp.exp(n),   "n²/eⁿ",       "0"),
]
for expr, label, expected in seqs:
    L = sp.limit(expr, n, sp.oo)
    print(f"   lim {label:12s} = {str(L):8s}   (αναμενόμενο {expected})")

# Η sympy.limit δεν χειρίζεται τον παράγοντα (-1)ⁿ· δουλεύουμε με παρεμβολή:
# |(-1)ⁿ/n| = 1/n -> 0, άρα και (-1)ⁿ/n -> 0.
Labs = sp.limit(sp.Abs(1/n), n, sp.oo)
print(f"   lim {'(-1)ⁿ/n':12s} : κριτήριο παρεμβολής, lim |(-1)ⁿ/n| = lim 1/n = "
      f"{Labs}  =>  όριο 0   (αναμενόμενο 0)")

alt = (-1)**n / n
print()
print("   Πρώτοι 10 όροι:")
for expr, label in [(e_, l_) for e_, l_, _ in seqs] + [(alt, "(-1)ⁿ/n")]:
    terms = [float(expr.subs(n, m)) for m in range(1, 11)]
    print(f"      {label:12s}: " + ", ".join(f"{t:+.5f}" for t in terms))

# Γράφημα: (1+1/n)^n -> e (οριζόντια ασύμπτωτος)
nv = np.arange(1, 201)
seq_e = (1 + 1.0 / nv) ** nv
plt.figure(figsize=(7, 4.2))
plt.plot(nv, seq_e, 'b.-', ms=3, lw=0.8, label=r'$(1+1/n)^n$')
plt.axhline(np.e, color='r', ls='--', label=f'e = {np.e:.6f} (οριζόντια ασύμπτωτος)')
plt.xlabel('n'); plt.ylabel(r'$a_n$')
plt.ylim(1.9, 2.9)
plt.title(r'Η ακολουθία $(1+1/n)^n$ συγκλίνει στο $e$')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
print(f"   a_200 = {seq_e[-1]:.8f}   e = {np.e:.8f}   διαφορά = {np.e - seq_e[-1]:.3e}")

# ── Β. Γεωμετρικές σειρές ───────────────────────────────────────────────────
print()
print("Β. ΓΕΩΜΕΤΡΙΚΕΣ ΣΕΙΡΕΣ  Σ_{n=0}^∞ rⁿ = 1/(1-r)")
print("-" * 60)
plt.figure(figsize=(7, 4.2))
Nmax = 30
idx = np.arange(Nmax + 1)
for r, col in [(0.5, 'tab:blue'), (2/3, 'tab:orange'), (-0.5, 'tab:green')]:
    rr = sp.Rational(1, 2) if r == 0.5 else (sp.Rational(2, 3) if r > 0 else sp.Rational(-1, 2))
    exact = sp.summation(rr**k, (k, 0, sp.oo))
    S = np.cumsum(r ** idx)                      # μερικά αθροίσματα με np.cumsum
    print(f"   r = {str(rr):5s}: sympy = {str(exact):5s} = {float(exact):.8f}   "
          f"1/(1-r) = {1/(1-r):.8f}   S_30 = {S[-1]:.8f}")
    plt.plot(idx, S, 'o-', ms=3, color=col, label=f'r = {rr}')
    plt.axhline(float(exact), color=col, ls='--', lw=1)
print("   (αναμενόμενα: 2, 3, 2/3)")
plt.xlabel('N'); plt.ylabel(r'$S_N=\sum_{n=0}^{N} r^n$')
plt.title('Μερικά αθροίσματα γεωμετρικών σειρών (διακεκομμένες: τα όρια)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()

# ── Γ. Κριτήριο λόγου (αριθμητικά, N=50) και σειρά Basel ────────────────────
print()
print("Γ. ΑΡΙΘΜΗΤΙΚΟ ΚΡΙΤΗΡΙΟ ΛΟΓΟΥ  (N = 50 όροι)")
print("-" * 60)
N = 50
ns = np.arange(1, N + 1)
a1 = np.array([math.factorial(int(m)) / float(m) ** int(m) for m in ns])   # n!/nⁿ
a2 = np.array([float(m) ** int(m) / math.factorial(int(m)) for m in ns])   # nⁿ/n!
r1 = a1[1:] / a1[:-1]
r2 = a2[1:] / a2[:-1]
print(f"   Σ n!/nⁿ : λόγος a_50/a_49 = {r1[-1]:.10f}   -> 1/e = {1/np.e:.10f}"
      f"   ({'< 1  ΣΥΓΚΛΙΝΕΙ' if r1[-1] < 1 else '>= 1'})")
print(f"             S_50 = {a1.sum():.10f}")
print(f"   Σ nⁿ/n! : λόγος a_50/a_49 = {r2[-1]:.10f}   -> e   = {np.e:.10f}"
      f"   ({'> 1  ΑΠΟΚΛΙΝΕΙ' if r2[-1] > 1 else '<= 1'})")
print(f"             S_50 = {a2.sum():.6e}  (εκρήγνυται)")

print()
print("   Σ 1/n² -> π²/6 : σφάλμα σε log-log")
Nb = 2000
nb = np.arange(1, Nb + 1)
Sb = np.cumsum(1.0 / nb**2)
err = np.pi**2 / 6 - Sb
for N0 in (10, 100, 1000):
    print(f"      S_{N0:<5d} = {Sb[N0-1]:.10f}   σφάλμα = {err[N0-1]:.3e}"
          f"   (1/N = {1.0/N0:.3e})")
slope = np.polyfit(np.log(nb[9:]), np.log(err[9:]), 1)[0]
print(f"      Κλίση της log(σφάλμα) ως προς log(N) = {slope:.4f}   (αναμενόμενη -1)")

plt.figure(figsize=(7, 4.2))
plt.loglog(nb, err, 'b-', label=r'$\pi^2/6-S_N$')
plt.loglog(nb, 1.0 / nb, 'r--', label=r'$1/N$ (κλίση $-1$)')
plt.xlabel('N'); plt.ylabel('σφάλμα')
plt.title(r'Σφάλμα της $\sum 1/n^2$ — ευθεία κλίσης $-1$ σε log-log')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()

# ── Δ. Τηλεσκοπική σειρά έναντι αποκλίνουσας ────────────────────────────────
print()
print("Δ. ΤΗΛΕΣΚΟΠΙΚΕΣ ΣΕΙΡΕΣ")
print("-" * 60)
Nt = 1000
nt = np.arange(1, Nt + 1)
tele = np.cumsum(1.0 / np.sqrt(nt) - 1.0 / np.sqrt(nt + 1))   # -> 1
logs = np.cumsum(np.log(1.0 + 1.0 / nt))                      # = ln(N+1), αποκλίνει
print("   Σ (1/√n - 1/√(n+1))  = 1 - 1/√(N+1)  ->  1")
for N0 in (10, 100, 1000):
    print(f"      S_{N0:<5d} = {tele[N0-1]:.10f}   κλειστός τύπος 1-1/√(N+1) = "
          f"{1 - 1/np.sqrt(N0+1):.10f}")
print("   Σ ln(1+1/n) = ln(N+1)  ->  +∞  (αποκλίνει)")
for N0 in (10, 100, 1000):
    print(f"      S_{N0:<5d} = {logs[N0-1]:.10f}   ln({N0+1}) = {np.log(N0+1):.10f}")
print(f"   Μέγιστη απόκλιση από τους κλειστούς τύπους: "
      f"{max(np.max(np.abs(tele - (1 - 1/np.sqrt(nt+1)))), np.max(np.abs(logs - np.log(nt+1)))):.2e}")
print("   ΣΥΜΠΕΡΑΣΜΑ: και στις δύο οι όροι τείνουν στο 0, αλλά μόνο η πρώτη")
print("   έχει φραγμένα μερικά αθροίσματα. Το a_n -> 0 ΔΕΝ αρκεί για σύγκλιση.")

plt.figure(figsize=(7.5, 4.5))
plt.plot(nt, tele, 'b-', lw=2, label=r'$\sum(1/\sqrt{n}-1/\sqrt{n+1})\to 1$')
plt.axhline(1, color='b', ls=':', lw=1)
plt.plot(nt, logs, 'r-', lw=2, label=r'$\sum\ln(1+1/n)=\ln(N+1)\to\infty$')
plt.xscale('log')
plt.xlabel('N (λογαριθμικός άξονας)'); plt.ylabel(r'$S_N$')
plt.title('Τηλεσκοπική που συγκλίνει έναντι τηλεσκοπικής που αποκλίνει')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
