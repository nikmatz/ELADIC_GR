-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 9 · Θεώρημα Weierstrass · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ9.Weierstrass

theorem extreme_values (f : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) (hf : ContinuousOn f (Set.Icc a b)) :
    ∃ xmin ∈ Set.Icc a b, ∃ xmax ∈ Set.Icc a b,
      ∀ x ∈ Set.Icc a b, f xmin ≤ f x ∧ f x ≤ f xmax := by
  obtain ⟨xmin, hmin, hmin'⟩ := isCompact_Icc.exists_isMinOn (Set.nonempty_Icc.mpr hab) hf
  obtain ⟨xmax, hmax, hmax'⟩ := isCompact_Icc.exists_isMaxOn (Set.nonempty_Icc.mpr hab) hf
  exact ⟨xmin, hmin, xmax, hmax, fun x hx => ⟨isMinOn_iff.mp hmin' x hx, isMaxOn_iff.mp hmax' x hx⟩⟩

end Κεφ9.Weierstrass
