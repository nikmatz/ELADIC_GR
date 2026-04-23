# ============================================================
# python_matrices.py
# Κεφάλαιο 2 — Άλγεβρα Πινάκων
# Βιβλίο: Μαθηματικά Ι · ΑΣΠΕΤΕ
# Συγγραφέας: Ν. Ματζάκος
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   numpy         → αριθμητικές πράξεις πινάκων
#   numpy.linalg  → det, inv, solve, LU (μέσω scipy)
#   scipy.linalg  → αποσύνθεση LU
#   sympy         → ακριβείς υπολογισμοί, inv με κλάσματα
#   matplotlib    → οπτικοποίηση (heatmaps πινάκων)
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   A @ B                  → γινόμενο πινάκων
#   A.T                    → ανάστροφος
#   np.linalg.det(A)       → ορίζουσα
#   np.linalg.inv(A)       → αντίστροφος
#   np.linalg.solve(A, b)  → λύση Ax = b
#   scipy.linalg.lu(A)     → αποσύνθεση LU
# ============================================================

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sympy import Matrix, Rational, pprint, latex

print("=" * 55)
print(" Κεφάλαιο 2: Άλγεβρα Πινάκων — Python")
print("=" * 55)

# ── Α. Βασικές Πράξεις ───────────────────────────────────
print("\n── Α. Βασικές Πράξεις Πινάκων ──")

A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]], dtype=float)

B = np.array([[2, -1, 0],
              [1,  3, 1],
              [0,  2, 4]], dtype=float)

print("A =\n", A)
print("\nB =\n", B)
print("\nA + B =\n", A + B)
print("\n3·A =\n", 3 * A)
print("\nA @ B =\n", A @ B)
print("\nB @ A =\n", B @ A)
print("\nΑ@B ≠ B@A  →  μη αντιμεταθετικότητα ✓")

# ── Β. Ανάστροφος, Ορίζουσα, Αντίστροφος ─────────────────
print("\n── Β. Ανάστροφος & Αντίστροφος ──")

print("Aᵀ =\n", A.T)
print(f"\ndet(A) = {np.linalg.det(A):.4f}")
print(f"det(B) = {np.linalg.det(B):.4f}")
print(f"det(A@B) = {np.linalg.det(A @ B):.4f}")
print(f"det(A)·det(B) = {np.linalg.det(A)*np.linalg.det(B):.4f}  ✓")

A_inv = np.linalg.inv(A)
print("\nA⁻¹ =\n", np.round(A_inv, 6))
print("\nA @ A⁻¹ =\n", np.round(A @ A_inv, 10))
print("≈ I₃ ✓")

# Ακριβής υπολογισμός με SymPy
print("\nΑκριβής A⁻¹ (SymPy με κλάσματα):")
A_sym = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
pprint(A_sym.inv())

# ── Γ. Επίλυση Ax = b ─────────────────────────────────────
print("\n── Γ. Επίλυση Ax = b ──")

b = np.array([1, 2, 3], dtype=float)
print(f"b = {b}")

# Μέθοδος 1: np.linalg.solve (αριθμητικά)
x_num = np.linalg.solve(A, b)
print(f"\nx = np.linalg.solve(A,b) = {np.round(x_num, 6)}")
print(f"Επαλήθευση A@x = {np.round(A @ x_num, 10)}  ✓")

# Μέθοδος 2: SymPy (ακριβής)
b_sym = Matrix([1, 2, 3])
x_sym = A_sym.solve(b_sym)
print("\nx (SymPy, ακριβής):")
pprint(x_sym)

# ── Δ. Αποσύνθεση LU ──────────────────────────────────────
print("\n── Δ. Αποσύνθεση LU ──")
# A = P·L·U  (scipy επιστρέφει P, L, U)

P, L, U = la.lu(A)

print("P (permutation) =\n", P)
print("\nL (lower triangular) =\n", np.round(L, 6))
print("\nU (upper triangular) =\n", np.round(U, 6))
print("\nΕπαλήθευση P@L@U = A:")
print(np.round(P @ L @ U, 10))
print("✓")

# ── Ε. Ειδικοί Πίνακες ────────────────────────────────────
print("\n── Ε. Ειδικοί Πίνακες ──")

S = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]], dtype=float)

print("S (συμμετρικός): S = Sᵀ →", np.allclose(S, S.T), "✓")
print(f"det(S) = {np.linalg.det(S):.4f}  (> 0 → θετικά ορισμένος)")

# Τα ιδιοδιανύσματα ενός συμμετρικού πίνακα είναι ορθογώνια
eigvals, eigvecs = np.linalg.eigh(S)   # eigh για συμμετρικό
print(f"\nΙδιοτιμές S: {np.round(eigvals, 4)}")
print("Ιδιοδιανύσματα (στήλες Q):\n", np.round(eigvecs, 4))
print(f"Όλες > 0  →  S θετικά ορισμένος ✓")

# ── Στ. Γραφική Απεικόνιση ────────────────────────────────
print("\n── Στ. Γραφική Απεικόνιση ──")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Άλγεβρα Πινάκων — Heatmaps", fontsize=13, fontweight='bold')

def plot_matrix(ax, M, title, fmt=".1f"):
    """Εμφάνιση πίνακα ως heatmap με τιμές."""
    im = ax.imshow(M, cmap='RdBu_r', aspect='auto',
                   norm=mcolors.TwoSlopeNorm(vcenter=0))
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, format(M[i, j], fmt),
                    ha='center', va='center', fontsize=9,
                    color='black' if abs(M[i,j]) < max(abs(M.max()), abs(M.min()))*0.7 else 'white')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)

plot_matrix(axes[0], A,     "Πίνακας A")
plot_matrix(axes[1], A @ B, "A · B")
plot_matrix(axes[2], L,     "L (αποσύνθεση LU)")

plt.tight_layout()
plt.savefig("matrix_algebra.png", dpi=120, bbox_inches='tight')
print("Το διάγραμμα αποθηκεύτηκε: matrix_algebra.png")
plt.show()
