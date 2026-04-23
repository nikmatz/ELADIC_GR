# ============================================================
# python_eigen.py
# Κεφάλαιο 5 — Ιδιοτιμές και Ιδιοδιανύσματα
# Βιβλίο: Μαθηματικά Ι · ΑΣΠΕΤΕ
# Συγγραφέας: Ν. Ματζάκος
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   numpy         → αριθμητικές ιδιοτιμές (eig, eigh)
#   sympy         → ακριβείς ιδιοτιμές, χαρακτηριστικό πολυώνυμο
#   matplotlib    → οπτικοποίηση ιδιοδιανυσμάτων, σύγκλιση power iteration
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   np.linalg.eig(A)     → ιδιοτιμές + ιδιοδιανύσματα (αριθμητικά)
#   np.linalg.eigh(A)    → για συμμετρικούς (ακριβέστερο)
#   sympy.Matrix.eigenvals()   → ιδιοτιμές (ακριβείς, με πολλαπλότητες)
#   sympy.Matrix.eigenvects()  → ιδιοτιμές + ιδιοδιανύσματα (ακριβείς)
#   sympy.Matrix.charpoly()    → χαρακτηριστικό πολυώνυμο
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix, Symbol, factor, expand, pprint, Rational

print("=" * 55)
print(" Κεφάλαιο 5: Ιδιοτιμές & Ιδιοδιανύσματα — Python")
print("=" * 55)

# ── Α. Χαρακτηριστικό Πολυώνυμο & Ιδιοτιμές (SymPy) ─────
print("\n── Α. Χαρακτηριστικό Πολυώνυμο ──")

A_sym = Matrix([[4, 1],
                [2, 3]])
lam = Symbol('lambda')

char_poly = A_sym.charpoly(lam)
print("det(A - λI) =", factor(char_poly.as_expr()))

evals = A_sym.eigenvals()
print(f"Ιδιοτιμές: {evals}")  # {τιμή: πολλαπλότητα}

print("\nΑκριβής ανάλυση (eigenvects):")
for eigval, mult, eigvecs in A_sym.eigenvects():
    print(f"  λ = {eigval}  (αλγεβρ. πολλαπλότητα = {mult})")
    for v in eigvecs:
        print(f"    ιδιοδιάνυσμα: ", end="")
        pprint(v.T)

# Επαλήθευση: trace = Σλ, det = Πλ
eigenvalues_list = list(evals.keys())
print(f"\nTrace(A) = {A_sym.trace()}  =  Σλᵢ = {sum(eigenvalues_list)}  ✓")
print(f"det(A)   = {A_sym.det()}    =  Πλᵢ = {eigenvalues_list[0]*eigenvalues_list[1]}  ✓")

# ── Β. NumPy — Αριθμητική Επίλυση ────────────────────────
print("\n── Β. Αριθμητικές Ιδιοτιμές (NumPy) ──")

A = np.array([[4, 1], [2, 3]], dtype=float)
vals, vecs = np.linalg.eig(A)

print(f"Ιδιοτιμές: {np.round(vals, 6)}")
for i, (lv, v) in enumerate(zip(vals, vecs.T)):
    print(f"  λ{i+1} = {lv:.4f}  →  v{i+1} = {np.round(v, 6)}")
    residual = A @ v - lv * v
    print(f"    Επαλήθευση Av - λv = {np.round(residual, 10)}  ✓")

# ── Γ. Πίνακας 3×3 ───────────────────────────────────────
print("\n── Γ. Πίνακας 3×3 ──")

C = np.array([[2, 0, 0],
              [1, 3, 0],
              [0, 1, 2]], dtype=float)

C_sym = Matrix([[2, 0, 0],
                [1, 3, 0],
                [0, 1, 2]])

cvals, cvecs = np.linalg.eig(C)
print(f"Ιδιοτιμές C: {np.round(np.sort(cvals), 6)}")
print(f"Trace(C) = {np.trace(C):.1f} = Σλᵢ = {np.sum(cvals):.4f}  ✓")
print(f"det(C)   = {np.linalg.det(C):.1f} = Πλᵢ = {np.prod(cvals):.4f}  ✓")

# ── Δ. Διαγωνοποίηση A = PDP⁻¹ ───────────────────────────
print("\n── Δ. Διαγωνοποίηση A = PDP⁻¹ ──")

vals_a, P = np.linalg.eig(A)
D = np.diag(vals_a)
A_recon = P @ D @ np.linalg.inv(P)
print(f"A (ανακατασκευή από PDP⁻¹):\n{np.round(A_recon, 8)}")
print(f"Ισότητα με A: {np.allclose(A_recon, A)}  ✓")

# Δύναμη πίνακα μέσω διαγωνοποίησης: A^10
n = 10
A_pow = P @ np.diag(vals_a**n) @ np.linalg.inv(P)
A_pow_direct = np.linalg.matrix_power(A, n)
print(f"\nA¹⁰ (μέσω PD¹⁰P⁻¹):\n{np.round(A_pow, 4)}")
print(f"Ισότητα με A^10 άμεσα: {np.allclose(A_pow, A_pow_direct)}  ✓")

# ── Ε. Power Iteration (Μέθοδος Δύναμης) ──────────────────
print("\n── Ε. Μέθοδος Δύναμης (Power Iteration) ──")

def power_iteration(A, num_iter=30):
    """Εκτιμά τη μεγαλύτερη ιδιοτιμή με επαναληπτική μέθοδο."""
    n = A.shape[0]
    b = np.random.rand(n); b /= np.linalg.norm(b)
    estimates = []
    for _ in range(num_iter):
        b_new = A @ b
        lam_est = np.dot(b, A @ b)
        estimates.append(lam_est)
        b = b_new / np.linalg.norm(b_new)
    return b, lam_est, estimates

np.random.seed(42)
eigvec_est, eigval_est, history = power_iteration(A)
print(f"Μεγαλύτερη ιδιοτιμή (power iter): {eigval_est:.8f}")
print(f"Ακριβής τιμή:                      {max(vals):.8f}  ✓")

# ── Στ. Αλυσίδα Markov ────────────────────────────────────
print("\n── Στ. Αλυσίδα Markov ──")

M = np.array([[0.8, 0.3],
              [0.2, 0.7]])

M_vals, M_vecs = np.linalg.eig(M)
print(f"Ιδιοτιμές M: {np.round(M_vals, 6)}")

# Στάσιμη κατανομή = ιδιοδιάνυσμα για λ=1
idx = np.argmax(np.abs(M_vals - 1) < 1e-10)
pi_vec = np.abs(M_vecs[:, idx])
pi_vec /= pi_vec.sum()
print(f"Στάσιμη κατανομή π = {np.round(pi_vec, 6)}")
print(f"Επαλήθευση Mπ = π: {np.allclose(M @ pi_vec, pi_vec)}  ✓")

# ── Ζ. Γραφική Απεικόνιση ──────────────────────────────────
print("\n── Ζ. Γραφική Απεικόνιση ──")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Ιδιοτιμές & Ιδιοδιανύσματα", fontsize=13, fontweight='bold')

# --- Ιδιοδιανύσματα στο επίπεδο ---
ax1 = axes[0]
ax1.set_title("Ιδιοδιανύσματα $A$")
colors = ['royalblue', 'tomato']
for i, (lv, v) in enumerate(zip(vals, vecs.T)):
    v_n = v / np.linalg.norm(v)
    ax1.annotate('', xy=v_n, xytext=(0,0),
                 arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5))
    ax1.annotate(f'$v_{i+1}$\n$\\lambda_{i+1}={lv:.0f}$',
                 v_n, fontsize=9, color=colors[i],
                 xytext=(5,5), textcoords='offset points')
ax1.set_xlim(-1.5,1.5); ax1.set_ylim(-1.5,1.5)
ax1.axhline(0,color='k',lw=0.5); ax1.axvline(0,color='k',lw=0.5)
ax1.grid(True,alpha=0.3); ax1.set_aspect('equal')
ax1.set_xlabel('x'); ax1.set_ylabel('y')

# --- Power iteration σύγκλιση ---
ax2 = axes[1]
ax2.set_title("Σύγκλιση Power Iteration")
ax2.plot(history, 'royalblue', lw=2)
ax2.axhline(max(vals), color='tomato', ls='--', lw=1.5, label=f'$\\lambda_{{max}}={max(vals):.1f}$')
ax2.set_xlabel("Επανάληψη"); ax2.set_ylabel("Εκτίμηση $\\lambda$")
ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3)

# --- Markov σύγκλιση ---
ax3 = axes[2]
ax3.set_title("Σύγκλιση Αλυσίδας Markov")
state = np.array([1.0, 0.0])
states = [state]
for _ in range(25):
    state = M @ state
    states.append(state)
states = np.array(states)
ax3.plot(states[:,0], 'royalblue', lw=2, label='Κατάσταση 1')
ax3.plot(states[:,1], 'tomato', lw=2, label='Κατάσταση 2')
ax3.axhline(pi_vec[0], color='royalblue', ls='--', lw=1, alpha=0.6)
ax3.axhline(pi_vec[1], color='tomato', ls='--', lw=1, alpha=0.6)
ax3.set_xlabel("Βήματα"); ax3.set_ylabel("Πιθανότητα")
ax3.legend(fontsize=9); ax3.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig("eigenvalues.png", dpi=120, bbox_inches='tight')
print("Το διάγραμμα αποθηκεύτηκε: eigenvalues.png")
plt.show()
