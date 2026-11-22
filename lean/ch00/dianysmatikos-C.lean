-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · ℂ ως Διανυσματικός Χώρος · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.DianysmatikosC
open Complex

theorem dimension : Module.finrank ℝ ℂ = 2 := finrank_real_complex

theorem basis_one_I : ⇑basisOneI = ![1, I] := coe_basisOneI

theorem unique_repr (z : ℂ) : ∃! p : ℝ × ℝ, (p.1 : ℂ) * 1 + p.2 * I = z := by
  refine ⟨(z.re, z.im), ?_, ?_⟩
  · simp
  · rintro ⟨a, b⟩ h
    subst h
    simp

end Κεφ0.DianysmatikosC
