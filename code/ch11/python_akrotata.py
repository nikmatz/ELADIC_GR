# ============================================================
# python_akrotata.py
# Κεφάλαιο 11 — Μονοτονία και Ακρότατα, SymPy & Matplotlib
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy       → κρίσιμα σημεία, κριτήριο 2ης παραγώγου
#   numpy       → πίνακας προσήμων σε πυκνό πλέγμα (np.sign)
#   matplotlib  → γραφήματα, σκίαση διαστημάτων (axvspan)
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.solve(diff(f,x), x)   → κρίσιμα σημεία
#   sympy.diff(f, x, 2)         → δεύτερη παράγωγος
#   sympy.lambdify(x, f)        → συμβολικό → αριθμητικό
#   np.sign(...)                → πρόσημο σε πυκνό πλέγμα
#   ax.axvspan(a, b, ...)       → χρωματισμός διαστήματος
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, solve, simplify, lambdify, sqrt, limit

x = symbols('x', real=True)

print("=" * 58)
print(" Κεφάλαιο 11: Μονοτονία, Ακρότατα, Κυρτότητα — Python")
print("=" * 58)

# Η συνάρτηση του βιβλίου
f  = x**3 - 3*x**2 - 9*x + 5
f1 = diff(f, x)
f2 = diff(f, x, 2)

print(f"\nf(x)   = {f}")
print(f"f'(x)  = {f1}")
print(f"f''(x) = {f2}")

# ── Α. Κρίσιμα σημεία — κριτήριο 2ης παραγώγου ────────────
print("\n── Α. Κρίσιμα σημεία και κριτήριο 2ης παραγώγου ──")

crits = sorted(solve(f1, x))
print(f"  Κρίσιμα σημεία (f'=0): x = {crits}   (αναμ. [-1, 3])")

kinds = {}
for xi in crits:
    yi  = f.subs(x, xi)
    f2i = f2.subs(x, xi)
    if f2i > 0:
        kind = "τοπικό ΕΛΑΧΙΣΤΟ"
    elif f2i < 0:
        kind = "τοπικό ΜΕΓΙΣΤΟ"
    else:
        kind = "ασαφές (χρειάζεται άλλο κριτήριο)"
    kinds[xi] = kind
    print(f"    x = {xi}:  f = {yi},  f'' = {f2i}  →  {kind}")
print("  Αναμενόμενα: f(-1)=10 τοπικό μέγιστο, f(3)=-22 τοπικό ελάχιστο.")

f_num  = lambdify(x, f,  'numpy')
f1_num = lambdify(x, f1, 'numpy')
f2_num = lambdify(x, f2, 'numpy')

xs = np.linspace(-4, 6, 800)

# Κοινό διάγραμμα f, f', f''
plt.figure(figsize=(7.5, 5))
plt.plot(xs, f_num(xs),  'b-',  lw=2,   label=r"$f(x)=x^3-3x^2-9x+5$")
plt.plot(xs, f1_num(xs), 'g--', lw=1.8, label=r"$f'(x)=3x^2-6x-9$")
plt.plot(xs, f2_num(xs), 'r:',  lw=1.8, label=r"$f''(x)=6x-6$")
plt.axhline(0, color='k', lw=0.6)
for xi in crits:
    plt.scatter([float(xi)], [float(f.subs(x, xi))], color='darkred', s=70, zorder=5)
    plt.annotate(f"({xi}, {f.subs(x, xi)})", (float(xi), float(f.subs(x, xi))),
                 textcoords="offset points", xytext=(8, 8), fontsize=9)
plt.grid(alpha=0.3); plt.legend(fontsize=9); plt.xlabel('x')
plt.title("Κεφ. 11 — f, f′ και f″ σε κοινό διάγραμμα")
plt.tight_layout()

# ── Β. Αυτόματος πίνακας προσήμων της f' (np.sign + axvspan) ──
print("\n── Β. Πίνακας προσήμων της f' (np.sign σε πυκνό πλέγμα) ──")

xg = np.linspace(-4, 6, 2001)
s1_all = np.sign(f1_num(xg))

# Αγνοούμε τα σημεία όπου f'=0 (εκεί το πρόσημο δεν ορίζεται) και
# κοιτάμε μόνο διαδοχικά ΜΗ μηδενικά πρόσημα.
nz = np.nonzero(s1_all)[0]
s1 = s1_all[nz]
xnz = xg[nz]
changes = np.where(np.diff(s1) != 0)[0]
print(f"  Αλλαγές προσήμου της f' μεταξύ x = "
      f"{np.round(xnz[changes], 3)} και {np.round(xnz[changes + 1], 3)}"
      f"   (αναμ. ≈ -1 και 3)")
for i in changes:
    before, after = s1[i], s1[i + 1]
    if before > 0 > after:
        kind = "τοπικό ΜΕΓΙΣΤΟ (+ → -)"
    elif before < 0 < after:
        kind = "τοπικό ΕΛΑΧΙΣΤΟ (- → +)"
    else:
        kind = "χωρίς αλλαγή"
    print(f"    στο διάστημα [{xnz[i]:7.3f}, {xnz[i+1]:7.3f}]: "
          f"{before:+.0f} → {after:+.0f}  ⇒  {kind}")

# Διαστήματα μονοτονίας από τα κρίσιμα σημεία
bounds = [-np.inf] + [float(c) for c in crits] + [np.inf]
mono = []
print("\n  Πίνακας μονοτονίας (δοκιμαστική τιμή ανά διάστημα):")
for a, b in zip(bounds[:-1], bounds[1:]):
    # δοκιμαστική τιμή μέσα στο διάστημα
    if np.isinf(a):
        t = b - 1.0
    elif np.isinf(b):
        t = a + 1.0
    else:
        t = 0.5*(a + b)
    v = float(f1_num(t))
    lab = "ΑΥΞΟΥΣΑ" if v > 0 else ("ΦΘΙΝΟΥΣΑ" if v < 0 else "—")
    mono.append((a, b, v > 0))
    astr = "-∞" if np.isinf(a) else f"{a:g}"
    bstr = "+∞" if np.isinf(b) else f"{b:g}"
    print(f"    ({astr}, {bstr}):  x={t:6.2f} → f'={v:9.3f}  "
          f"{'(+)' if v > 0 else '(-)'}  {lab}")

# Γράφημα f με χρωματισμένα τα διαστήματα αύξησης/μείωσης
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, f_num(xs), 'b-', lw=2.2, label=r"$f(x)=x^3-3x^2-9x+5$")
for a, b, inc in mono:
    a_p = max(a, xs[0]); b_p = min(b, xs[-1])
    ax.axvspan(a_p, b_p, color=('#bfe6bf' if inc else '#f6c6c6'), alpha=0.55)
ax.axhline(0, color='k', lw=0.6)
for xi in crits:
    ax.scatter([float(xi)], [float(f.subs(x, xi))], color='darkred', s=70, zorder=5)
ax.set_xlim(xs[0], xs[-1])
ax.grid(alpha=0.3); ax.set_xlabel('x')
ax.set_title("Μονοτονία: πράσινο = αύξουσα (f′>0), κόκκινο = φθίνουσα (f′<0)")
ax.legend(fontsize=9)
plt.tight_layout()

# ── Γ. Σημεία καμπής και κυρτότητα / κοιλότητα ────────────
print("\n── Γ. Σημεία καμπής — κυρτές / κοίλες περιοχές ──")

infl = sorted(solve(f2, x))
print(f"  Σημεία καμπής (f''=0): x = {infl}   (αναμ. [1])")
for xi in infl:
    print(f"    x = {xi}:  f = {f.subs(x, xi)}   (αναμ. f(1) = -6)")

# Επαλήθευση αλλαγής προσήμου της f'' σε κάθε σημείο καμπής
eps = 0.1
for xi in infl:
    left  = float(f2_num(float(xi) - eps))
    right = float(f2_num(float(xi) + eps))
    print(f"    f''({float(xi)-eps:g}) = {left:+.3f},  "
          f"f''({float(xi)+eps:g}) = {right:+.3f}  →  "
          f"αλλαγή προσήμου: {left*right < 0}")

# Διαστήματα κυρτότητας
cb = [-np.inf] + [float(i) for i in infl] + [np.inf]
conv = []
print("\n  Πίνακας κυρτότητας:")
for a, b in zip(cb[:-1], cb[1:]):
    if np.isinf(a):
        t = b - 1.0
    elif np.isinf(b):
        t = a + 1.0
    else:
        t = 0.5*(a + b)
    v = float(f2_num(t))
    lab = "ΚΥΡΤΗ (convex)" if v > 0 else ("ΚΟΙΛΗ (concave)" if v < 0 else "—")
    conv.append((a, b, v > 0))
    astr = "-∞" if np.isinf(a) else f"{a:g}"
    bstr = "+∞" if np.isinf(b) else f"{b:g}"
    print(f"    ({astr}, {bstr}):  x={t:6.2f} → f''={v:9.3f}  "
          f"{'(+)' if v > 0 else '(-)'}  {lab}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, f_num(xs), 'b-', lw=2.2, label=r"$f(x)$")
ax.plot(xs, f2_num(xs), 'r:', lw=1.5, label=r"$f''(x)$")
for a, b, cx in conv:
    a_p = max(a, xs[0]); b_p = min(b, xs[-1])
    ax.axvspan(a_p, b_p, color=('#c9d8f2' if cx else '#f2e3c0'), alpha=0.65)
ax.axhline(0, color='k', lw=0.6)
for xi in infl:
    ax.scatter([float(xi)], [float(f.subs(x, xi))], color='purple', s=80, zorder=5)
    ax.annotate(f"καμπή ({xi}, {f.subs(x, xi)})",
                (float(xi), float(f.subs(x, xi))),
                textcoords="offset points", xytext=(10, -18), fontsize=9)
ax.set_xlim(xs[0], xs[-1])
ax.grid(alpha=0.3); ax.set_xlabel('x')
ax.set_title("Κυρτότητα: μπλε = κυρτή (f″>0), ώχρα = κοίλη (f″<0)")
ax.legend(fontsize=9)
plt.tight_layout()

# ── Δ. Θεώρημα Μέσης Τιμής ────────────────────────────────
print("\n── Δ. Θεώρημα Μέσης Τιμής ──")


def mvt(expr, a, b, name):
    """Βρίσκει c στο (a,b) με f'(c) = (f(b)-f(a))/(b-a) και το επιστρέφει."""
    fa, fb = expr.subs(x, a), expr.subs(x, b)
    slope = simplify((fb - fa)/(b - a))
    d = diff(expr, x)
    sols = [s for s in solve(d - slope, x) if s.is_real and a < s < b]
    print(f"\n  {name} στο [{a}, {b}]")
    print(f"    f({a}) = {fa},  f({b}) = {fb}")
    print(f"    κλίση χορδής = {slope}  ( = {float(slope):.6f} )")
    print(f"    f'(x) = {d}")
    print(f"    λύσεις της f'(c) = κλίση στο ({a},{b}): c = {sols}")
    return fa, fb, slope, sols


# (i) f(x) = x^2 - x + 1 στο [1,4]  → c = 5/2
g1 = x**2 - x + 1
a1, b1 = 1, 4
fa1, fb1, m1, cs1 = mvt(g1, a1, b1, "f(x) = x² - x + 1")
c1 = cs1[0]
gc1 = g1.subs(x, c1)
chord1   = fa1 + m1*(x - a1)
tangent1 = gc1 + m1*(x - c1)
print(f"    c = {c1}  (αναμ. 5/2),  f(c) = {gc1}  (αναμ. 19/4)")
print(f"    χορδή:      y = {chord1.expand()}   (αναμ. 4x - 3)")
print(f"    εφαπτομένη: y = {tangent1.expand()}   (αναμ. 4x - 21/4)")
print(f"    κλίσεις: χορδή {diff(chord1, x)}, εφαπτομένη {diff(tangent1, x)} "
      f"→ παράλληλες: {simplify(diff(chord1, x) - diff(tangent1, x)) == 0}")

g1n = lambdify(x, g1, 'numpy')
ch1n = lambdify(x, chord1, 'numpy')
tg1n = lambdify(x, tangent1, 'numpy')
xx1 = np.linspace(0.5, 4.5, 400)

# (ii) g(x) = sqrt(x) στο [0,4]  → c = 1
g2 = sqrt(x)
a2, b2 = 0, 4
fa2, fb2, m2, cs2 = mvt(g2, a2, b2, "g(x) = √x")
c2 = cs2[0]
gc2 = g2.subs(x, c2)
chord2   = fa2 + m2*(x - a2)
tangent2 = gc2 + m2*(x - c2)
print(f"    c = {c2}  (αναμ. 1),  g(c) = {gc2}  (αναμ. 1)")
print(f"    χορδή:      y = {chord2.expand()}   (αναμ. x/2)")
print(f"    εφαπτομένη: y = {tangent2.expand()}   (αναμ. x/2 + 1/2)")
print(f"    κλίσεις: χορδή {diff(chord2, x)}, εφαπτομένη {diff(tangent2, x)} "
      f"→ παράλληλες: {simplify(diff(chord2, x) - diff(tangent2, x)) == 0}")

# Σχόλιο για την υπόθεση της παραγωγισιμότητας στα ΑΚΡΑ
dg2 = diff(g2, x)
lim0 = limit(dg2, x, 0, '+')
print(f"\n    ΣΧΟΛΙΟ: g'(x) = {dg2}, και lim_(x→0+) g'(x) = {lim0}.")
print("    Η √x ΔΕΝ είναι παραγωγίσιμη στο άκρο x=0 (κατακόρυφη εφαπτομένη).")
print("    Το Θ.Μ.Τ. ζητά συνέχεια στο ΚΛΕΙΣΤΟ [a,b] και παραγωγισιμότητα")
print("    μόνο στο ΑΝΟΙΚΤΟ (a,b) — άρα οι υποθέσεις ΙΣΧΥΟΥΝ και εδώ,")
print("    γι' αυτό και βρέθηκε c = 1 στο (0,4).")

g2n = lambdify(x, g2, 'numpy')
ch2n = lambdify(x, chord2, 'numpy')
tg2n = lambdify(x, tangent2, 'numpy')
xx2 = np.linspace(0, 4.5, 400)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axes[0]
ax.plot(xx1, g1n(xx1), 'b-', lw=2.2, label=r"$f(x)=x^2-x+1$")
ax.plot(xx1, ch1n(xx1), 'g--', lw=1.8, label="χορδή A(1,1)–B(4,13)")
ax.plot(xx1, tg1n(xx1), 'r-.', lw=1.8, label=f"εφαπτομένη στο c={c1}")
ax.scatter([a1, b1], [float(fa1), float(fb1)], color='green', s=60, zorder=5)
ax.scatter([float(c1)], [float(gc1)], color='red', s=80, zorder=5)
ax.axvline(float(c1), color='0.6', lw=0.8, ls=':')
ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlabel('x')
ax.set_title(f"Θ.Μ.Τ. στο [1,4]:  c = {c1}")

ax = axes[1]
ax.plot(xx2, g2n(xx2), 'b-', lw=2.2, label=r"$g(x)=\sqrt{x}$")
ax.plot(xx2, ch2n(xx2), 'g--', lw=1.8, label="χορδή A(0,0)–B(4,2)")
ax.plot(xx2, tg2n(xx2), 'r-.', lw=1.8, label=f"εφαπτομένη στο c={c2}")
ax.scatter([a2, b2], [float(fa2), float(fb2)], color='green', s=60, zorder=5)
ax.scatter([float(c2)], [float(gc2)], color='red', s=80, zorder=5)
ax.axvline(float(c2), color='0.6', lw=0.8, ls=':')
ax.set_ylim(-0.3, 3.2)
ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlabel('x')
ax.set_title(f"Θ.Μ.Τ. για √x στο [0,4]:  c = {c2}\n(μη παραγωγίσιμη στο άκρο x=0)")

plt.tight_layout()
plt.show()

print("\nΟλοκληρώθηκε το Κεφάλαιο 11.")
