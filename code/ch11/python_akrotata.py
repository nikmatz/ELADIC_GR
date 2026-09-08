# ============================================================
# python_akrotata.py
# Κεφάλαιο 11 — Μονοτονία, Κυρτότητα και Ακρότατα
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, solve, lambdify, simplify

x = symbols('x', real=True)

print("=" * 58)
print(" Κεφάλαιο 11: Μονοτονία, Κυρτότητα και Ακρότατα")
print("=" * 58)

f = x**3 - 3*x**2 - 9*x + 2
f1 = diff(f, x)
f2 = diff(f, x, 2)

print(f"\nf(x)  = {f}")
print(f"f'(x) = {f1}")
print(f"f''(x)= {f2}")

crits = solve(f1, x)
print(f"\nΚρίσιμα σημεία (f'=0): x = {crits}")
for xi in crits:
    yi  = f.subs(x,xi)
    f2i = f2.subs(x,xi)
    kind = "min" if f2i > 0 else ("max" if f2i < 0 else "?")
    print(f"  x={xi}: f={yi},  f''={f2i}  -> Τοπικό {kind}")

infl = solve(f2, x)
print(f"\nΣημεία καμπής (f''=0): x = {infl}")
for xi in infl:
    print(f"  x={xi}: f={f.subs(x,xi)}")

f_num  = lambdify(x, f,  'numpy')
f1_num = lambdify(x, f1, 'numpy')
f2_num = lambdify(x, f2, 'numpy')
xs = np.linspace(-4, 7, 500)

fig, axes = plt.subplots(1,3,figsize=(13,4))
for ax, fn, lbl, col in zip(axes,
    [f_num,f1_num,f2_num],
    [r'$f$',r"$f'$",r"$f''$"],
    ['b','g','r']):
    ax.plot(xs,fn(xs),f'{col}-',lw=2,label=lbl)
    ax.axhline(0,color='k',lw=0.5,ls='--')
    ax.set_xlabel('x'); ax.legend(); ax.grid(True,alpha=0.3)
plt.suptitle('Μελέτη f(x)=x³-3x²-9x+2')
plt.tight_layout(); plt.savefig('ch10_study.png',dpi=100)
print("\n  Γράφημα: ch10_study.png")
print("\n✓ Ολοκληρώθηκε.")

# ============================================================
# ΣΥΜΠΛΗΡΩΜΑ — Πίνακας προσήμου της f' με np.sign
# ============================================================
import numpy as np
from sympy import symbols, diff, lambdify, solve

x = symbols('x')
f  = x**3 - 3*x**2 - 9*x + 2
f1 = diff(f, x)

crit = sorted([float(r) for r in solve(f1, x) if r.is_real])
print("Κρίσιμα σημεία:", crit)

f1n = lambdify(x, f1, 'numpy')
xs  = np.linspace(-4, 7, 1101)
s   = np.sign(f1n(xs))              # +1 / 0 / -1

# Εντοπίζουμε τις αλλαγές προσήμου: εκεί βρίσκονται τα ακρότατα.
changes = np.where(np.diff(s) != 0)[0]
print("Αλλαγές προσήμου της f' κοντά στα x =",
      np.round(xs[changes], 3))

for i in changes:
    before, after = s[i], s[i+1]
    kind = "τοπικό μέγιστο" if before > 0 > after else \
           "τοπικό ελάχιστο" if before < 0 < after else "χωρίς ακρότατο"
    print(f"  x ~ {xs[i]:6.3f}: {before:+.0f} -> {after:+.0f}  =>  {kind}")

# Διαστήματα μονοτονίας από το πρόσημο:
print("f' > 0 (αύξουσα) στο ποσοστό δείγματος:",
      f"{100*np.mean(s > 0):.1f}%")
