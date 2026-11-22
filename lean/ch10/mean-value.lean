-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 10 · Θεώρημα Μέσης Τιμής (Lagrange) · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ10.MesiTimi

theorem mean_value (f f' : ℝ → ℝ) (a b : ℝ) (hab : a < b) (hfc : ContinuousOn f (Set.Icc a b))
    (hff' : ∀ x ∈ Set.Ioo a b, HasDerivAt f (f' x) x) :
    ∃ c ∈ Set.Ioo a b, f' c = (f b - f a) / (b - a) := by
  set k := (f b - f a) / (b - a) with hk
  have hcont : ContinuousOn (fun x => f x - k * (x - a)) (Set.Icc a b) := by
    exact hfc.sub (continuousOn_const.mul (continuousOn_id.sub continuousOn_const))
  have hderiv : ∀ x ∈ Set.Ioo a b, HasDerivAt (fun x => f x - k * (x - a)) (f' x - k) x := by
    intro x hx
    have h := ((hasDerivAt_id x).sub_const a).const_mul k
    rw [mul_one] at h
    exact (hff' x hx).sub h
  have hends : f a - k * (a - a) = f b - k * (b - a) := by
    rw [sub_self, mul_zero, sub_zero, hk, div_mul_cancel₀ _ (sub_ne_zero.mpr hab.ne')]
    ring
  obtain ⟨c, hc, hc0⟩ := exists_hasDerivAt_eq_zero hab hcont hends hderiv
  exact ⟨c, hc, by linarith⟩

theorem rolle_from_mvt (f f' : ℝ → ℝ) (a b : ℝ) (hab : a < b)
    (hfc : ContinuousOn f (Set.Icc a b)) (hff' : ∀ x ∈ Set.Ioo a b, HasDerivAt f (f' x) x)
    (hfab : f a = f b) : ∃ c ∈ Set.Ioo a b, f' c = 0 := by
  obtain ⟨c, hc, h⟩ := mean_value f f' a b hab hfc hff'
  rw [hfab, sub_self, zero_div] at h
  exact ⟨c, hc, h⟩

end Κεφ10.MesiTimi
