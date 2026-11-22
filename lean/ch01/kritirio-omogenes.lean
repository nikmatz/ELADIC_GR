-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Κριτήριο Γραμμικής Εξάρτησης μέσω Γραμμικού Συστήματος · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ1.KritirioOmogenes

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem dependent_iff {k : ℕ} (v : Fin k → V) :
    ¬ LinearIndependent ℝ v ↔ ∃ x : Fin k → ℝ, x ≠ 0 ∧ ∑ i, x i • v i = 0 := by
  rw [Fintype.not_linearIndependent_iff]
  constructor
  · rintro ⟨g, hg, i, hi⟩
    exact ⟨g, fun h => hi (by simp [h]), hg⟩
  · rintro ⟨x, hx, hsum⟩
    obtain ⟨i, hi⟩ := Function.ne_iff.mp hx
    exact ⟨x, hsum, i, hi⟩

theorem dependent_of_zero {k : ℕ} (v : Fin k → V) (i : Fin k) (h : v i = 0) :
    ¬ LinearIndependent ℝ v := by
  intro hli
  exact hli.ne_zero i h

end Κεφ1.KritirioOmogenes
