# ============================================================
# python_efarm_par.py
# Κεφάλαιο 12 — Εφαρμογές Παραγώγου: SymPy & SciPy
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy  → όρια (L'Hôpital), κρίσιμα σημεία
#   scipy  → optimize.minimize_scalar, optimize.newton
#   numpy / matplotlib → γραφήματα σύγκλισης και βελτιστοποίησης
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.limit(f, x, a)                  → όριο (dir='+' για πλευρικό)
#   sympy.solve(diff(f, x), x)            → κρίσιμα σημεία
#   scipy.optimize.minimize_scalar(f, ...)→ αριθμητική ελαχιστοποίηση
#   scipy.optimize.newton(f, x0, fprime)  → αριθμητική ρίζα (Newton)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, newton
from sympy import (symbols, diff, limit, solve, simplify, lambdify, together,
                   fraction, exp, log, sin, oo, nan, zoo, S)

x = symbols('x')

print("=" * 58)
print(" Κεφάλαιο 12: Εφαρμογές Παραγώγου — Python")
print("=" * 58)

# ── Α. Κανόνας L'Hôpital — βήμα βήμα ──────────────────────
print("\n── Α. Κανόνας L'Hôpital ──")

INDET = (S.Zero, oo, -oo, zoo, nan)


def _form(num, den, pt, direction):
    """Επιστρέφει (lim αριθμητή, lim παρονομαστή) στο σημείο pt."""
    ln = limit(num, x, pt, direction)
    ld = limit(den, x, pt, direction)
    return ln, ld


def _is_indeterminate(ln, ld):
    """0/0 ή ±oo/±oo ;"""
    zero_zero = (ln == 0 and ld == 0)
    inf_inf = (ln in (oo, -oo, zoo) and ld in (oo, -oo, zoo))
    return zero_zero or inf_inf


def lhopital(num, den, pt, direction='+-', max_steps=5, label=""):
    """Εφαρμόζει L'Hôpital βήμα-βήμα και τυπώνει κάθε ενδιάμεσο βήμα."""
    print(f"\n  {label}")
    ln, ld = _form(num, den, pt, direction)
    print(f"    Βήμα 0: ({num}) / ({den})")
    print(f"            αριθμητής → {ln},  παρονομαστής → {ld}")
    if not _is_indeterminate(ln, ld):
        val = limit(num/den, x, pt, direction)
        print(f"            δεν είναι απροσδιόριστη μορφή· τιμή = {val}")
        return val
    form = "0/0" if ln == 0 else "±oo/±oo"
    print(f"            απροσδιόριστη μορφή {form} → εφαρμόζουμε L'Hôpital")

    n_k, d_k = num, den
    for k in range(1, max_steps + 1):
        n_raw, d_raw = diff(n_k, x), diff(d_k, x)
        print(f"    Βήμα {k}: ({n_raw}) / ({d_raw})")
        # Αλγεβρική αναδιάταξη του πηλίκου — αλλιώς μορφές όπως
        # (1/x)/(-1/x^2) θα έμεναν "oo/oo" επ' άπειρον.
        n_k, d_k = fraction(together(simplify(n_raw/d_raw)))
        if (n_k, d_k) != (n_raw, d_raw):
            print(f"            μετά από απλοποίηση: ({n_k}) / ({d_k})")
        ln, ld = _form(n_k, d_k, pt, direction)
        print(f"            αριθμητής → {ln},  παρονομαστής → {ld}")
        if not _is_indeterminate(ln, ld):
            val = limit(n_k/d_k, x, pt, direction)
            print(f"            η αοριστία ΑΡΘΗΚΕ · όριο = {val}")
            return val
        form = "0/0" if ln == 0 else "±oo/±oo"
        print(f"            πάλι {form} → συνεχίζουμε")
    print(f"    Δεν άρθηκε σε {max_steps} βήματα.")
    return None


# (1) lim_{x→0} (e^x - 1 - x)/x^2      [0/0]  → 1/2
v1 = lhopital(exp(x) - 1 - x, x**2, 0,
              label="(1) lim_(x→0) (e^x - 1 - x)/x^2      [μορφή 0/0]")
d1 = limit((exp(x) - 1 - x)/x**2, x, 0)
print(f"    sympy.limit = {d1}   (αναμ. 1/2)   L'Hôpital βήμα-βήμα: {v1}"
      f"   συμφωνία: {simplify(v1 - d1) == 0}")

# (2) lim_{x→0+} x·ln x                [0·(-oo)] → 0
print("\n  (2) lim_(x→0+) x·ln(x)      [μορφή 0 · (-oo)]")
print(f"    x → {limit(x, x, 0, '+')},  ln(x) → {limit(log(x), x, 0, '+')}")
print("    Μετατροπή σε πηλίκο: x·ln x = ln(x) / (1/x)   [μορφή -oo/oo]")
v2 = lhopital(log(x), 1/x, 0, direction='+',
              label="(2β) lim_(x→0+) ln(x)/(1/x)")
d2 = limit(x*log(x), x, 0, '+')
print(f"    sympy.limit = {d2}   (αναμ. 0)   L'Hôpital βήμα-βήμα: {v2}"
      f"   συμφωνία: {simplify(v2 - d2) == 0}")

# (3) lim_{x→+oo} x^2/e^x              [oo/oo] → 0
v3 = lhopital(x**2, exp(x), oo,
              label="(3) lim_(x→+oo) x^2/e^x      [μορφή oo/oo]")
d3 = limit(x**2/exp(x), x, oo)
print(f"    sympy.limit = {d3}   (αναμ. 0)   L'Hôpital βήμα-βήμα: {v3}"
      f"   συμφωνία: {simplify(v3 - d3) == 0}")
print("    Ερμηνεία: η e^x κυριαρχεί κάθε πολυωνύμου.")

# (4) lim_{x→0} (1/x - 1/sin x)        [oo - oo] → 0
print("\n  (4) lim_(x→0) (1/x - 1/sin x)      [μορφή oo - oo]")
expr4 = 1/x - 1/sin(x)
num4, den4 = fraction(together(expr4))
print(f"    Κοινός παρονομαστής: ({num4}) / ({den4})")
v4 = lhopital(num4, den4, 0, label="(4β) lim_(x→0) (sin x - x)/(x·sin x)")
d4 = limit(expr4, x, 0)
print(f"    sympy.limit = {d4}   (αναμ. 0)   L'Hôpital βήμα-βήμα: {v4}"
      f"   συμφωνία: {simplify(v4 - d4) == 0}")

# ── Β. Βελτιστοποίηση: ορθογώνιο με περίμετρο P = 20 ──────
print("\n── Β. Ορθογώνιο περιμέτρου P=20: E(x) = x(10-x) ──")

E = x*(10 - x)
dE = diff(E, x)
sol_E = solve(dE, x)
xE = sol_E[0]
print(f"  E'(x) = {dE},  E'(x)=0 → x = {sol_E}   (αναμ. [5])")
print(f"  E''(x) = {diff(E, x, 2)} < 0 → ΜΕΓΙΣΤΟ")
print(f"  E_max = E({xE}) = {E.subs(x, xE)}   (αναμ. 25 — τετράγωνο 5×5)")

# Αριθμητική επαλήθευση: ελαχιστοποιούμε το -E
E_num = lambdify(x, E, 'numpy')
resE = minimize_scalar(lambda t: -E_num(t), bounds=(0, 10), method='bounded')
print(f"  scipy.minimize_scalar(-E): x = {resE.x:.8f}, E = {-resE.fun:.8f}")
print(f"  Απόκλιση από το x=5: {abs(resE.x - float(xE)):.2e}")

# ── Γ. Ανοικτό δοχείο V=32: S(x) = x^2 + 128/x ────────────
print("\n── Γ. Ανοικτό δοχείο (χωρίς καπάκι) V=32 ──")

xp = symbols('x', positive=True)
h_box = 32/xp**2                      # ύψος από τον όγκο: x^2·h = 32
S_expr = (xp**2 + 4*xp*h_box).expand()  # βάση + 4 πλευρές (χωρίς καπάκι)
print(f"  ύψος h = 32/x²,  S(x) = x² + 4xh = {S_expr}")
print(f"  Ταυτίζεται με x² + 128/x: "
      f"{simplify(S_expr - (xp**2 + 128/xp)) == 0}")

dS = simplify(diff(S_expr, xp))
sol_S = [s for s in solve(dS, xp) if s.is_real and s > 0]
xS = sol_S[0]
print(f"  S'(x) = {dS}")
print(f"  S'(x)=0 → x = {sol_S}   (αναμ. [4])")
print(f"  S''(x) = {simplify(diff(S_expr, xp, 2))},  "
      f"S''({xS}) = {diff(S_expr, xp, 2).subs(xp, xS)} > 0 → ΕΛΑΧΙΣΤΟ")
print(f"  S_min = S({xS}) = {S_expr.subs(xp, xS)}   (αναμ. 48)")
print(f"  ύψος h = {h_box.subs(xp, xS)}   (αναμ. 2)")
print(f"  Έλεγχος όγκου: x²·h = {(xp**2*h_box).subs(xp, xS)}   (αναμ. 32)")

S_num = lambdify(xp, S_expr, 'numpy')
resS = minimize_scalar(S_num, bounds=(0.5, 20), method='bounded')
print(f"  scipy.minimize_scalar(S): x = {resS.x:.8f}, S = {resS.fun:.8f}")
print(f"  Απόκλιση από το x=4: {abs(resS.x - float(xS)):.2e}")

xs_S = np.linspace(1.0, 12.0, 500)
plt.figure(figsize=(7, 4.5))
plt.plot(xs_S, S_num(xs_S), 'b-', lw=2.2, label=r"$S(x)=x^2+\dfrac{128}{x}$")
plt.scatter([float(xS)], [float(S_expr.subs(xp, xS))], color='red', s=90,
            zorder=5, label=f"ελάχιστο: x={xS}, S={S_expr.subs(xp, xS)}")
plt.axvline(float(xS), color='0.6', lw=0.9, ls=':')
plt.axhline(float(S_expr.subs(xp, xS)), color='0.6', lw=0.9, ls=':')
plt.ylim(0, 250); plt.grid(alpha=0.3); plt.legend()
plt.xlabel('πλευρά βάσης x'); plt.ylabel('επιφάνεια S')
plt.title("Ανοικτό δοχείο V=32: ελάχιστη επιφάνεια")
plt.tight_layout()

# ── Δ. Newton-Raphson: x^3 - x - 2 = 0, x0 = 1.5 ──────────
print("\n── Δ. Newton-Raphson για x³ - x - 2 = 0 (x₀ = 1.5) ──")

fN_sym = x**3 - x - 2
dfN_sym = diff(fN_sym, x)
fN = lambdify(x, fN_sym, 'numpy')
dfN = lambdify(x, dfN_sym, 'numpy')
print(f"  f(x) = {fN_sym},  f'(x) = {dfN_sym}")

# Ρίζα αναφοράς με scipy
root_scipy, info = newton(fN, x0=1.5, fprime=dfN, tol=1e-14, full_output=True)
print(f"  scipy.optimize.newton: ρίζα = {root_scipy:.13f}  "
      f"({info.iterations} επαναλήψεις)")
print(f"  f(ρίζα) = {fN(root_scipy):.3e}")

x0 = 1.5
iterates = [x0]
errors = [abs(x0 - root_scipy)]
print("\n  i        x_i                   f(x_i)          |x_i - ρίζα|   "
      "σωστά ψηφία")
print(f"  0   {x0:.13f}   {fN(x0):+.6e}   {errors[0]:.3e}")
xi = x0
for i in range(1, 6):
    xi = xi - fN(xi)/dfN(xi)
    err = abs(xi - root_scipy)
    iterates.append(xi)
    errors.append(err)
    digits = (f"{-np.log10(err):5.1f}" if err > 0 else " (ακριβές)")
    print(f"  {i}   {xi:.13f}   {fN(xi):+.6e}   {err:.3e}   {digits}")

print(f"\n  Ρίζα (Newton, 5 βήματα): {iterates[-1]:.13f}")
print(f"  Ρίζα (scipy.newton):     {root_scipy:.13f}")
print(f"  Διαφορά: {abs(iterates[-1] - root_scipy):.3e}")
print("  Αναμενόμενη ρίζα: 1.5213797068046")
print("  Τετραγωνική σύγκλιση: τα σωστά δεκαδικά ψηφία περίπου")
print("  διπλασιάζονται σε κάθε βήμα (≈3.4 → 7.0 → 14.1).")

# Γράφημα σύγκλισης: εφαπτόμενες και διαδοχικές προσεγγίσεις
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
xs_N = np.linspace(1.2, 1.75, 400)
ax.plot(xs_N, fN(xs_N), 'b-', lw=2.2, label=r"$f(x)=x^3-x-2$")
ax.axhline(0, color='k', lw=0.8)
cols = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
for k in range(min(3, len(iterates) - 1)):
    xk = iterates[k]
    yk = fN(xk)
    # εφαπτόμενη στο (x_k, f(x_k)) — τέμνει τον άξονα στο x_{k+1}
    tang = yk + dfN(xk)*(xs_N - xk)
    ax.plot(xs_N, tang, '--', lw=1.2, color=cols[k],
            label=f"εφαπτόμενη στο $x_{k}$ = {xk:.6f}")
    ax.plot([xk, xk], [0, yk], ':', lw=1, color=cols[k])
    ax.scatter([xk], [yk], color=cols[k], s=45, zorder=5)
    ax.scatter([iterates[k + 1]], [0], color=cols[k], s=45, marker='v', zorder=5)
ax.scatter([root_scipy], [0], color='k', s=80, marker='*', zorder=6,
           label=f"ρίζα ≈ {root_scipy:.7f}")
ax.set_ylim(-0.6, 1.2); ax.set_xlim(1.2, 1.75)
ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlabel('x')
ax.set_title("Newton: κάθε εφαπτόμενη δίνει την επόμενη προσέγγιση")

ax = axes[1]
nz = [(i, e) for i, e in enumerate(errors) if e > 0]
ax.semilogy([i for i, _ in nz], [e for _, e in nz], 'o-', lw=2, color='crimson')
for i, e in nz:
    ax.annotate(f"{e:.1e}", (i, e), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
ax.set_xlabel('επανάληψη i'); ax.set_ylabel(r'$|x_i - r|$  (λογαριθμική)')
ax.grid(alpha=0.3, which='both')
ax.set_title("Σφάλμα σε ημιλογαριθμική κλίμακα\n(η καμπύλωση ⇒ τετραγωνική σύγκλιση)")

plt.tight_layout()
plt.show()

print("\nΟλοκληρώθηκε το Κεφάλαιο 12.")
