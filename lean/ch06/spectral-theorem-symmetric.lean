-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 6 · Φασματικό Θεώρημα · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ6.FasmatikoTheorima
open Matrix

variable {n : ℕ}

theorem real_eigenvalues (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    A.charpoly = ∏ i, (Polynomial.X - Polynomial.C (hA.eigenvalues i)) :=
  hA.charpoly_eq

theorem eigenvectors (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) (j : Fin n) :
    A *ᵥ ⇑(hA.eigenvectorBasis j) = hA.eigenvalues j • ⇑(hA.eigenvectorBasis j) :=
  hA.mulVec_eigenvectorBasis j

theorem orthonormal_eigenvectors (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    Orthonormal ℝ hA.eigenvectorBasis :=
  hA.eigenvectorBasis.orthonormal

theorem spectral (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.IsHermitian) :
    ∃ Q : Matrix (Fin n) (Fin n) ℝ, Qᵀ * Q = 1 ∧ Q * Qᵀ = 1 ∧
      A = Q * diagonal hA.eigenvalues * Qᵀ := by
  refine ⟨hA.eigenvectorUnitary, ?_, ?_, ?_⟩
  · rw [← conjTranspose_eq_transpose_of_trivial, ← star_eq_conjTranspose]
    exact Unitary.coe_star_mul_self hA.eigenvectorUnitary
  · rw [← conjTranspose_eq_transpose_of_trivial, ← star_eq_conjTranspose]
    exact Unitary.coe_mul_star_self hA.eigenvectorUnitary
  · conv_lhs => rw [hA.spectral_theorem]
    rw [Unitary.conjStarAlgAut_apply, star_eq_conjTranspose,
      conjTranspose_eq_transpose_of_trivial, RCLike.ofReal_real_eq_id, Function.id_comp]

end Κεφ6.FasmatikoTheorima
