-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 13 · Λογαριθμικός Κανόνας Ολοκλήρωσης · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ13.LogarithmikosKanonas

theorem log_rule (f f' : ℝ → ℝ) (I : Set ℝ) (hf : ∀ x ∈ I, HasDerivAt f (f' x) x)
    (hne : ∀ x ∈ I, f x ≠ 0) :
    ∀ x ∈ I, HasDerivAt (fun x => Real.log |f x|) (f' x / f x) x := by
  intro x hx
  have habs : (fun x => Real.log |f x|) = fun x => Real.log (f x) := by
    funext y
    exact Real.log_abs (f y)
  rw [habs]
  have hlog : HasDerivAt Real.log (f x)⁻¹ (f x) := Real.hasDerivAt_log (hne x hx)
  have hfx := hf x hx
  have h := hlog.comp x hfx
  rw [div_eq_mul_inv, mul_comm]
  exact h

theorem tan_rule (x : ℝ) (hx : Real.cos x ≠ 0) :
    HasDerivAt (fun x => Real.log |Real.cos x|) (-Real.sin x / Real.cos x) x := by
  exact log_rule Real.cos (fun x => -Real.sin x) {y | Real.cos y ≠ 0}
    (fun y _ => Real.hasDerivAt_cos y) (fun y hy => hy) x hx

end Κεφ13.LogarithmikosKanonas
