# ============================================================
# python_ttl.py
# Κεφάλαιο 15 — Θεμελιώδες Θεώρημα Ολοκληρωτικού Λογισμού (ΘΘΟΛ)
# Ματζάκος, Ν. (2026). Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   sympy       → συμβολική συνάρτηση εμβαδού και κανόνας Leibniz
#   numpy       → σωρευτικό άθροισμα λωρίδων (np.cumsum)
#   scipy       → αριθμητική ολοκλήρωση αναφοράς (quad)
#   matplotlib  → κοινά διαγράμματα και fill_between
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   sympy.integrate(f, (t, 0, x)) → F(x) = ∫₀ˣ f(t)dt
#   sympy.diff(F, x)              → F'(x) = f(x)          [ΘΘΟΛ-Α]
#   sympy.Integral(f, (t, a(x), b(x))) → ανεκτέλεστο ολοκλήρωμα
#   sympy.diff(Integral, x)       → κανόνας Leibniz
#   scipy.integrate.quad(f, a, b) → αριθμητικό ολοκλήρωμα
#   numpy.cumsum(y) * h           → σωρευτικό άθροισμα λωρίδων
#   matplotlib.fill_between()     → σκίαση χωρίου
#
# Δραστηριότητα βιβλίου:
#   (α) F(x)=∫₀ˣf(t)dt για τρεις f· έλεγχος F'=f· f και F σε ΚΟΙΝΟ διάγραμμα
#   (β) np.cumsum για f(x)=sin x στο [0,2π] έναντι της ακριβούς 1-cos x
#   (γ) κανόνας Leibniz για d/dx ∫ₓ^(x²) sin(t²)dt + πεπερασμένες διαφορές
#   (δ) ΘΜΤ Ολοκληρωτικού Λογισμού για f(x)=x² στο [0,3]
# ============================================================

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

x, t = sp.symbols('x t', real=True)

print("=" * 62)
print(" Κεφάλαιο 15: Θεμελιώδες Θεώρημα Ολοκληρωτικού Λογισμού")
print("=" * 62)

# ── Α. (α) Η συνάρτηση εμβαδού F(x) = ∫₀ˣ f(t)dt ──────────
print("\n──── Α. (α) Συνάρτηση εμβαδού και έλεγχος F'(x) = f(x) ────")

cases = [("t²",        t**2,          (-2.0,  2.0)),
         ("sin t",     sp.sin(t),     (0.0,   2*np.pi)),
         ("1/(1+t²)",  1/(1 + t**2),  (-4.0,  4.0))]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
fig.suptitle("Μέρος Α του ΘΘΟΛ:  F(x) = ∫₀ˣ f(t)dt  και  F'(x) = f(x)",
             fontsize=12, fontweight='bold')

for ax, (name, ft, (lo, hi)) in zip(axes, cases):
    F = sp.simplify(sp.integrate(ft, (t, 0, x)))
    dF = sp.simplify(sp.diff(F, x))
    ok = sp.simplify(dF - ft.subs(t, x)) == 0
    print(f"  f(t) = {name:<10s} → F(x) = {F}")
    print(f"      F'(x) = {dF}     F'(x) - f(x) = 0 ; {ok}")

    # f και F στο ΙΔΙΟ διάγραμμα
    zz = np.linspace(lo, hi, 500)
    fn = sp.lambdify(x, ft.subs(t, x), 'numpy')
    Fn = sp.lambdify(x, F, 'numpy')
    ax.plot(zz, fn(zz) * np.ones_like(zz), 'b-', lw=2, label=f"f(x) = {name}")
    ax.plot(zz, Fn(zz) * np.ones_like(zz), 'r-', lw=2, label=f"F(x) = {F}")
    ax.axhline(0, color='k', lw=.6); ax.axvline(0, color='k', lw=.6)
    ax.set_xlabel("x"); ax.legend(fontsize=8); ax.grid(alpha=.3)

plt.tight_layout()
print("\n  [Αναμένεται: x**3/3 ,  1 - cos(x) ,  atan(x)]")

# ── Β. (β) Αριθμητική συνάρτηση εμβαδού με np.cumsum ──────
print("\n──── Β. (β) np.cumsum για f(x) = sin x στο [0, 2π] ────")

N = 4001
xs = np.linspace(0, 2*np.pi, N)          # ΟΛΟΚΛΗΡΟ το [0,2π]
h = xs[1] - xs[0]
y = np.sin(xs)

# Σωρευτικό άθροισμα λωρίδων (αριστερά άκρα) — ο απλούστερος τρόπος
F_left = np.concatenate(([0.0], np.cumsum(y[:-1]) * h))
# Σωρευτικό άθροισμα τραπεζίων — πιο ακριβές, ίδια ιδέα (np.cumsum)
F_trap = np.concatenate(([0.0], np.cumsum((y[:-1] + y[1:])/2) * h))
F_exact = 1 - np.cos(xs)                 # η ακριβής συνάρτηση εμβαδού

print(f"  Πλέγμα: {N} σημεία στο [0, 2π],  h = {h:.6f}")
print(f"  max|F_cumsum(αριστερά) - (1-cos x)| = "
      f"{np.max(np.abs(F_left - F_exact)):.3e}")
print(f"  max|F_cumsum(τραπέζια)  - (1-cos x)| = "
      f"{np.max(np.abs(F_trap - F_exact)):.3e}")

for x0 in (np.pi/2, np.pi, 3*np.pi/2, 2*np.pi):
    val = quad(np.sin, 0, x0)[0]
    print(f"    quad ∫₀^{x0:.4f} sin t dt = {val: .10f}   "
          f"ακριβές 1-cos = {1-np.cos(x0): .10f}")

# Πού μηδενίζεται η F' και τι συμβαίνει εκεί στην f;
dF = np.gradient(F_trap, xs)
k_max = int(np.argmax(F_trap))
print(f"\n  Μέγιστο της F: F({xs[k_max]:.6f}) = {F_trap[k_max]:.6f}"
      f"   [x = π ≈ {np.pi:.6f}, F(π) = 2]")
print(f"  Εκεί: F'(x) ≈ {dF[k_max]:.3e}  και  f(x) = sin(x) = "
      f"{np.sin(xs[k_max]):.3e}")
interior = slice(3, -3)
print(f"  max|F'(x) - sin x| στο εσωτερικό = "
      f"{np.max(np.abs(dF[interior] - y[interior])):.3e}")
# Τα σημεία όπου η αριθμητική F' αλλάζει πρόσημο:
# γνήσια αλλαγή προσήμου (γινόμενο < 0) — αγνοεί τον θόρυβο γύρω από το 0
sign_changes = xs[:-1][dF[:-1]*dF[1:] < 0]
print(f"  Αλλαγές προσήμου της F' (= μηδενικά της f) στο [0,2π]: "
      f"{np.round(sign_changes, 5)}   [αναμενόμενο x ≈ π = {np.pi:.5f}]")
print("  ⇒ Όπου F' = 0 η f μηδενίζεται· στο x = π η f αλλάζει από + σε -,")
print("    οπότε η F σταματά να αυξάνεται και εκεί έχει το μέγιστό της.")

fig, (axf, axF) = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True)
fig.suptitle("Μέρος Α αριθμητικά: η συσσώρευση των λωρίδων (np.cumsum)",
             fontsize=12, fontweight='bold')
axf.plot(xs, y, 'b-', lw=2, label="f(x) = sin x")
axf.fill_between(xs, 0, y, where=(y >= 0), color='green', alpha=.20,
                 label="θετική συνεισφορά")
axf.fill_between(xs, 0, y, where=(y < 0), color='red', alpha=.20,
                 label="αρνητική συνεισφορά")
axf.axhline(0, color='k', lw=.6); axf.axvline(np.pi, color='gray', ls=':')
axf.legend(fontsize=8); axf.grid(alpha=.3); axf.set_ylabel("f(x)")

axF.plot(xs, F_trap, 'r-', lw=2.4, label="F(x) με np.cumsum")
axF.plot(xs, F_exact, 'k--', lw=1.4, label="ακριβής F(x) = 1 - cos x")
axF.axvline(np.pi, color='gray', ls=':')
axF.plot([np.pi], [2], 'ko', ms=7)
axF.annotate("μέγιστο F(π) = 2\n(εκεί f = 0)", (np.pi, 2),
             textcoords="offset points", xytext=(12, -28), fontsize=9)
axF.axhline(0, color='k', lw=.6)
axF.set_xlabel("x"); axF.set_ylabel("F(x)")
axF.legend(fontsize=8); axF.grid(alpha=.3)
axF.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
axF.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
plt.tight_layout()

# ── Γ. (γ) Κανόνας Leibniz με sympy.Integral ──────────────
print("\n──── Γ. (γ) Κανόνας Leibniz: d/dx ∫ₓ^(x²) sin(t²) dt ────")

I_unev = sp.Integral(sp.sin(t**2), (t, x, x**2))
print(f"  Ανεκτέλεστο: {I_unev}")
D = sp.simplify(sp.diff(I_unev, x))
print(f"  sympy.diff → {D}")

# Ο τύπος f(b(x))·b'(x) - f(a(x))·a'(x)
bx, ax_ = x**2, x
rule = (sp.sin(t**2).subs(t, bx)*sp.diff(bx, x)
        - sp.sin(t**2).subs(t, ax_)*sp.diff(ax_, x))
print(f"  Τύπος f(b)b' - f(a)a' → {sp.simplify(rule)}")
print(f"  Διαφορά (πρέπει 0): {sp.simplify(D - rule)}")
print("  [ΑΝΑΜΕΝΟΜΕΝΟ 2x·sin(x⁴) - sin(x²)]")

# Αριθμητική επαλήθευση με κεντρικές πεπερασμένες διαφορές
print("\n  Επαλήθευση με πεπερασμένες διαφορές:")
D_num = sp.lambdify(x, D, 'numpy')
g = lambda s: np.sin(s**2)
print(f"  {'x₀':>6} {'πεπ. διαφορές':>18} {'συμβολικός τύπος':>18} "
      f"{'|διαφορά|':>12}")
for x0 in (0.7, 1.0, 1.3, 1.8):
    hh = 1e-5
    Ip = quad(g, x0 + hh, (x0 + hh)**2)[0]
    Im = quad(g, x0 - hh, (x0 - hh)**2)[0]
    num = (Ip - Im)/(2*hh)
    sym = float(D_num(x0))
    print(f"  {x0:>6.2f} {num:>18.9f} {sym:>18.9f} {abs(num-sym):>12.2e}")

# ── Δ. (δ) ΘΜΤ Ολοκληρωτικού Λογισμού ────────────────────
print("\n──── Δ. (δ) ΘΜΤ Ολοκληρωτικού Λογισμού: f(x)=x² στο [0,3] ────")

c = sp.symbols('c', positive=True)
a_i, b_i = 0, 3
I_val = sp.integrate(x**2, (x, a_i, b_i))
print(f"  ∫₀³ x² dx = {I_val}   [ΑΝΑΜΕΝΟΜΕΝΟ 9]")

sols = sp.solve(sp.Eq(c**2*(b_i - a_i), I_val), c)
print(f"  solve: f(c)·(b-a) = ∫ₐᵇf  ⇒  c = {sols}")
c_val = sols[0]
print(f"  Δεκτή λύση στο [0,3]: c = {c_val} ≈ {float(c_val):.10f}"
      f"   [ΑΝΑΜΕΝΟΜΕΝΟ √3 ≈ {np.sqrt(3):.10f}]")
h_rect = c_val**2
print(f"  Ύψος ορθογωνίου f(c) = {h_rect}")
print(f"  Εμβαδόν ορθογωνίου f(c)·(b-a) = {sp.simplify(h_rect*(b_i-a_i))}"
      f"   έναντι  ∫ₐᵇf = {I_val}")
print(f"  Διαφορά (πρέπει 0): {sp.simplify(h_rect*(b_i - a_i) - I_val)}")

cf, hf = float(c_val), float(h_rect)
zz = np.linspace(0, 3, 500)
plt.figure(figsize=(7.8, 5.2))
plt.fill_between(zz, 0, zz**2, color='steelblue', alpha=.35,
                 label="χωρίο κάτω από f(x)=x²   (εμβαδόν = 9)")
plt.fill_between([0, 3], 0, [hf, hf], color='crimson', alpha=.18,
                 label=f"ορθογώνιο ύψους f(c) = {hf:.0f}   (εμβαδόν = {hf*3:.0f})")
plt.plot(zz, zz**2, 'b-', lw=2.3)
plt.axhline(hf, color='crimson', lw=2, ls='--')
plt.vlines(cf, 0, hf, color='k', ls=':', lw=1.3)
plt.plot([cf], [hf], 'ko', ms=8, zorder=5)
plt.annotate(f"c = √3 ≈ {cf:.4f}", (cf, hf), textcoords="offset points",
             xytext=(12, -22), fontsize=10)
plt.xlim(0, 3); plt.ylim(0, 9.6)
plt.xlabel("x"); plt.ylabel("y")
plt.title("ΘΜΤ Ολοκληρωτικού Λογισμού: τα δύο εμβαδά είναι ίσα")
plt.legend(fontsize=9, loc='upper left'); plt.grid(alpha=.3)
plt.tight_layout()

print("\n✓ Ολοκληρώθηκε το Κεφάλαιο 15.")
plt.show()
