-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 13 · Κανόνας Γραμμικής Αντικατάστασης · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ13.GrammikiAntikatastasi

theorem linear_substitution (F f : ℝ → ℝ) (a b : ℝ) (ha : a ≠ 0)
    (hF : ∀ u, HasDerivAt F (f u) u) :
    ∀ x, HasDerivAt (fun x => 1 / a * F (a * x + b)) (f (a * x + b)) x := by
  intro x
  have h1 : HasDerivAt (fun x => a * x + b) a x := by
    simpa using ((hasDerivAt_id x).const_mul a).add_const b
  have hFx := hF (a * x + b)
  have h2 := hFx.comp x h1
  have h3 : HasDerivAt (fun x => 1 / a * F (a * x + b)) (1 / a * (f (a * x + b) * a)) x :=
    h2.const_mul (1 / a)
  convert h3 using 1
  field_simp

theorem exp_linear (a b : ℝ) (ha : a ≠ 0) :
    ∀ x, HasDerivAt (fun x => 1 / a * Real.exp (a * x + b)) (Real.exp (a * x + b)) x := by
  exact linear_substitution Real.exp Real.exp a b ha Real.hasDerivAt_exp

end Κεφ13.GrammikiAntikatastasi
