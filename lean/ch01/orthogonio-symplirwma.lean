-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Ιδιότητες Ορθογώνιου Συμπληρώματος · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ1.OrthogonioSymplirwma
open scoped InnerProductSpace
open Module Submodule

variable {n : ℕ}

theorem mem_iff (W : Submodule ℝ (EuclideanSpace ℝ (Fin n))) (v : EuclideanSpace ℝ (Fin n)) :
    v ∈ Wᗮ ↔ ∀ w ∈ W, ⟪w, v⟫_ℝ = 0 :=
  mem_orthogonal W v

theorem inter_zero (W : Submodule ℝ (EuclideanSpace ℝ (Fin n))) : W ⊓ Wᗮ = ⊥ :=
  inf_orthogonal_eq_bot W

theorem direct_sum (W : Submodule ℝ (EuclideanSpace ℝ (Fin n))) : IsCompl W Wᗮ :=
  isCompl_orthogonal W

theorem dim_sum (W : Submodule ℝ (EuclideanSpace ℝ (Fin n))) :
    finrank ℝ W + finrank ℝ Wᗮ = n := by
  rw [finrank_add_finrank_orthogonal, finrank_euclideanSpace_fin]

theorem double_perp (W : Submodule ℝ (EuclideanSpace ℝ (Fin n))) : Wᗮᗮ = W :=
  orthogonal_orthogonal W

end Κεφ1.OrthogonioSymplirwma
