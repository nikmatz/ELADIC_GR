# ============================================================
# python_diag.py
# Κεφάλαιο 6 — Διαγωνοποίηση Πινάκων
# Βιβλίο: Μαθηματικά Ι · ΑΣΠΕΤΕ
# Συγγραφέας: Ν. Ματζάκος
# ============================================================
#
# ΒΙΒΛΙΟΘΗΚΕΣ:
#   numpy         → eig, eigh, svd, matrix_power
#   scipy.linalg  → expm (εκθετική πίνακα)
#   sympy         → ακριβής διαγωνοποίηση, τετραγωνικές μορφές
#   matplotlib    → ελλείψεις τετραγωνικής μορφής, SVD γεωμετρία
#
# ΒΑΣΙΚΕΣ ΕΝΤΟΛΕΣ:
#   np.linalg.eig(A)         → διαγωνοποίηση (γενική)
#   np.linalg.eigh(S)        → για συμμετρικούς (ορθ. ιδιοδ.)
#   np.linalg.svd(A)         → SVD: A = UΣVᵀ
#   scipy.linalg.expm(A)     → εκθετική e^A
#   np.linalg.matrix_power   → Aⁿ άμεσα (για σύγκριση)
# ============================================================

import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from sympy import Matrix, symbols, pprint, sqrt, Rational, factor

print("=" * 55)
print(" Κεφάλαιο 6: Διαγωνοποίηση — Python")
print("=" * 55)

# ── Α. Διαγωνοποίηση A = PDP⁻¹ ───────────────────────────
print("\n── Α. Διαγωνοποίηση A = PDP⁻¹ ──")

A = np.array([[5, 4],
              [1, 2]], dtype=float)

vals, P = np.linalg.eig(A)
D = np.diag(vals)
P_inv = np.linalg.inv(P)

print(f"Ιδιοτιμές: {np.round(vals, 6)}")
print(f"P =\n{np.round(P, 6)}")
print(f"\nΕπαλήθευση PDP⁻¹ = A:")
print(np.round(P @ D @ P_inv, 8))
print(f"✓ {np.allclose(P @ D @ P_inv, A)}")

# Δύναμη μέσω διαγωνοποίησης
n_pow = 10
A_pow_diag   = P @ np.diag(vals**n_pow) @ P_inv
A_pow_direct = np.linalg.matrix_power(A, n_pow)
print(f"\nA¹⁰ (PD¹⁰P⁻¹):\n{np.round(A_pow_diag, 2)}")
print(f"Ισότητα με matrix_power: {np.allclose(A_pow_diag, A_pow_direct)}  ✓")

# ── Β. Μη Διαγωνοποιήσιμος — Έλεγχος ─────────────────────
print("\n── Β. Έλεγχος Διαγωνοποιησιμότητας ──")

def is_diagonalizable(A):
    """Ελέγχει αν ο A είναι διαγωνοποιήσιμος."""
    A_sym = Matrix(A.tolist())
    n = A.shape[0]
    total_geo = 0
    for lam, alg_mult in A_sym.eigenvals().items():
        geo_mult = len(A_sym.eigenvects()[
            [ev[0] for ev in A_sym.eigenvects()].index(lam)][2])
        total_geo += geo_mult
        print(f"  λ={lam}: αλγ.πολλ.={alg_mult}, γεωμ.πολλ.={geo_mult}")
    return total_geo == n

B = np.array([[3, 1], [0, 3]], dtype=float)   # Jordan block
C = np.array([[5, 4], [1, 2]], dtype=float)   # διαγωνοποιήσιμος

print("B (Jordan block):")
is_diagonalizable(B)
print(f"  → Διαγωνοποιήσιμος: {np.linalg.matrix_rank(B - 3*np.eye(2)) < 2}")

# ── Γ. Ορθογώνια Διαγωνοποίηση (Φασματικό Θεώρημα) ───────
print("\n── Γ. Ορθογώνια Διαγωνοποίηση — Συμμετρικός ──")

S = np.array([[4, 2],
              [2, 1]], dtype=float)
print(f"S = Sᵀ: {np.allclose(S, S.T)}  ✓")

# eigh → ορθοκανονικά ιδιοδιανύσματα εγγυημένα
eigvals, Q = np.linalg.eigh(S)
Lambda = np.diag(eigvals)

print(f"Ιδιοτιμές Λ: {np.round(eigvals, 6)}")
print(f"Q (ορθοκανονικός):\n{np.round(Q, 6)}")
print(f"QᵀQ = I: {np.allclose(Q.T @ Q, np.eye(2))}  ✓")
print(f"QΛQᵀ = S: {np.allclose(Q @ Lambda @ Q.T, S)}  ✓")

# ── Δ. SVD — Αποσύνθεση Μοναδιαίων Τιμών ──────────────────
print("\n── Δ. SVD: A = UΣVᵀ ──")

M = np.array([[1, 2, 0],
              [0, 1, 3],
              [1, 0, 1]], dtype=float)

U, sigma, Vt = np.linalg.svd(M)
Sigma = np.zeros_like(M)
np.fill_diagonal(Sigma, sigma)

print(f"Μοναδιαίες τιμές σ: {np.round(sigma, 4)}")
print(f"Επαλήθευση UΣVᵀ = M: {np.allclose(U @ Sigma @ Vt, M)}  ✓")
print(f"rank(M) = {np.linalg.matrix_rank(M)}  "
      f"(= αριθμός μη-μηδενικών σ: {np.sum(sigma > 1e-10)})")

# Προσέγγιση χαμηλής τάξης (rank-1)
M_rank1 = sigma[0] * np.outer(U[:,0], Vt[0,:])
print(f"\nΠροσέγγιση rank-1:")
print(np.round(M_rank1, 4))
print(f"Σφάλμα Frobenius: {np.linalg.norm(M - M_rank1):.4f}")

# ── Ε. Εκθετική Πίνακα e^A ────────────────────────────────
print("\n── Ε. Εκθετική Πίνακα e^(tA) ──")

# A = PDP⁻¹  →  e^A = P·diag(e^λᵢ)·P⁻¹
A2 = np.array([[0, -1], [1,  0]], dtype=float)   # στροφή: e^(tA) = R(t)
t = np.pi / 2   # π/2

eA_scipy = sla.expm(t * A2)
eA_diag  = np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t),  np.cos(t)]])   # αναλυτικά: R(π/2)

print(f"A = antisymmetric (γεννήτρας στροφής)")
print(f"e^(π/2·A) (scipy):\n{np.round(eA_scipy, 6)}")
print(f"R(π/2) (αναλυτικά):\n{np.round(eA_diag, 6)}")
print(f"Ισότητα: {np.allclose(eA_scipy, eA_diag)}  ✓")

# ── Στ. Τετραγωνική Μορφή & Ελλείψεις ────────────────────
print("\n── Στ. Τετραγωνικές Μορφές ──")

# Q(x,y) = xᵀSx: ελλείψεις = ισοϋψείς καμπύλες Q=const
theta = np.linspace(0, 2*np.pi, 400)
unit_circle = np.array([np.cos(theta), np.sin(theta)])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Διαγωνοποίηση — SVD & Τετραγωνικές Μορφές", fontsize=12, fontweight='bold')

# --- Ιδιοδιανύσματα συμμετρικού S ---
ax1 = axes[0]
ax1.set_title("Ιδιοδιανύσματα $S$ (ορθογώνια)")
colors = ['royalblue', 'tomato']
for i in range(2):
    v = Q[:, i] * eigvals[i] if eigvals[i] != 0 else Q[:, i] * 0.5
    ax1.annotate('', xy=Q[:,i], xytext=(0,0),
                 arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5))
    ax1.text(Q[0,i]*1.1, Q[1,i]*1.1,
             f'$q_{i+1}$, $\\lambda={eigvals[i]:.0f}$', fontsize=9, color=colors[i])
ax1.axhline(0,color='k',lw=0.5); ax1.axvline(0,color='k',lw=0.5)
ax1.set_xlim(-1.5,1.5); ax1.set_ylim(-1.5,1.5)
ax1.set_aspect('equal'); ax1.grid(True,alpha=0.3)

# --- SVD: εφαρμογή σε μοναδιαίο κύκλο ---
ax2 = axes[1]
ax2.set_title("SVD: Εικόνα μοναδιαίου κύκλου")
transformed = M[:2,:2] @ unit_circle   # 2D για απλότητα (πρώτη 2×2 υπο-αρ/ση)
ax2.plot(unit_circle[0], unit_circle[1], 'royalblue', lw=1.5, label='Αρχικός κύκλος')
ax2.plot(transformed[0], transformed[1], 'tomato', lw=2, label='$M\\cdot$κύκλος')
ax2.axhline(0,color='k',lw=0.5); ax2.axvline(0,color='k',lw=0.5)
ax2.set_aspect('equal'); ax2.grid(True,alpha=0.3)
ax2.legend(fontsize=8); ax2.set_title("SVD: Εικόνα κύκλου υπό $M_{2\\times2}$")

# --- Ισοϋψείς τετραγωνικής μορφής Q=xᵀSx ---
ax3 = axes[2]
ax3.set_title("Ισοϋψείς $Q(x,y)=x^\\top S x$")
xx, yy = np.meshgrid(np.linspace(-3,3,300), np.linspace(-3,3,300))
pts = np.stack([xx.ravel(), yy.ravel()])
Qvals = np.sum(pts * (S @ pts), axis=0).reshape(xx.shape)
cs = ax3.contour(xx, yy, Qvals, levels=[1,2,4,8,16], colors='royalblue')
ax3.clabel(cs, fontsize=8)
# Άξονες ιδιοδιανυσμάτων
for i, col in enumerate(['tomato','forestgreen']):
    ax3.annotate('', xy=Q[:,i]*2, xytext=(0,0),
                 arrowprops=dict(arrowstyle='->', color=col, lw=2))
ax3.axhline(0,color='k',lw=0.5); ax3.axvline(0,color='k',lw=0.5)
ax3.set_xlim(-3,3); ax3.set_ylim(-3,3)
ax3.set_aspect('equal'); ax3.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig("diagonalization.png", dpi=120, bbox_inches='tight')
print("Το διάγραμμα αποθηκεύτηκε: diagonalization.png")
plt.show()
