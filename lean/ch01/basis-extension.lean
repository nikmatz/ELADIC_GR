-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Επέκταση σε Βάση · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ1.EpektasiBasis
open Module Submodule

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem extend (S : Set V) (hS : LinearIndepOn ℝ id S) :
    ∃ B : Set V, S ⊆ B ∧ LinearIndepOn ℝ id B ∧ span ℝ B = ⊤ := by
  obtain ⟨B, -, hSB, hspan, hB⟩ := exists_linearIndepOn_extension hS (Set.subset_univ S)
  refine ⟨B, hSB, hB, ?_⟩
  simpa using hspan

end Κεφ1.EpektasiBasis
