-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 4 · Ισομορφισμός και Διάσταση · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ4.IsomorfismosDiastasi

variable {V W : Type*} [AddCommGroup V] [Module ℝ V] [AddCommGroup W] [Module ℝ W]

theorem iso_iff_dim [FiniteDimensional ℝ V] [FiniteDimensional ℝ W] :
    Nonempty (V ≃ₗ[ℝ] W) ↔ Module.finrank ℝ V = Module.finrank ℝ W :=
  FiniteDimensional.nonempty_linearEquiv_iff_finrank_eq

theorem iso_Rn [FiniteDimensional ℝ V] (n : ℕ) (h : Module.finrank ℝ V = n) :
    Nonempty (V ≃ₗ[ℝ] (Fin n → ℝ)) := by
  rw [iso_iff_dim, h, Module.finrank_fin_fun]

end Κεφ4.IsomorfismosDiastasi
