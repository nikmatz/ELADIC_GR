-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Συντεταγμένες ως προς Ορθοκανονική Βάση · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ1.Syntetagmenes
open scoped InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem coordinate {n : ℕ} (b : OrthonormalBasis (Fin n) ℝ V) (c : Fin n → ℝ) (j : Fin n) :
    ⟪∑ i, c i • b i, b j⟫_ℝ = c j := by
  rw [sum_inner]
  simp only [real_inner_smul_left]
  rw [Finset.sum_eq_single j]
  · rw [real_inner_self_eq_norm_sq, b.orthonormal.1 j]
    ring
  · intro i _ hij
    rw [b.orthonormal.2 hij, mul_zero]
  · intro hj
    exact absurd (Finset.mem_univ j) hj

theorem expansion {n : ℕ} (b : OrthonormalBasis (Fin n) ℝ V) (v : V) :
    v = ∑ i, ⟪v, b i⟫_ℝ • b i := by
  conv_lhs => rw [← b.sum_repr' v]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [real_inner_comm]

end Κεφ1.Syntetagmenes
