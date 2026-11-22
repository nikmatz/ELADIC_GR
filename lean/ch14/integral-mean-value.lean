-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 14 · Θεώρημα Μέσης Τιμής Ολοκληρωτικού Λογισμού · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ14.MesiTimi
open MeasureTheory

theorem integral_mean_value (f : ℝ → ℝ) (a b : ℝ) (hab : a < b)
    (hf : ContinuousOn f (Set.Icc a b)) :
    ∃ ξ ∈ Set.Icc a b, f ξ = 1 / (b - a) * ∫ x in a..b, f x := by
  -- Weierstrass: ελάχιστη τιμή m = f(x₁) και μέγιστη τιμή M = f(x₂) στο [a,b]
  obtain ⟨x₁, hx₁, hmin⟩ := isCompact_Icc.exists_isMinOn (Set.nonempty_Icc.mpr hab.le) hf
  obtain ⟨x₂, hx₂, hmax⟩ := isCompact_Icc.exists_isMaxOn (Set.nonempty_Icc.mpr hab.le) hf
  have hint : IntervalIntegrable f volume a b := hf.intervalIntegrable_of_Icc hab.le
  -- m (b - a) ≤ ∫ f: ολοκληρώνουμε την m ≤ f(x)
  have h1 : f x₁ * (b - a) ≤ ∫ x in a..b, f x := by
    have h := intervalIntegral.integral_mono_on hab.le (intervalIntegrable_const (c := f x₁)) hint
      (fun x hx => isMinOn_iff.mp hmin x hx)
    rw [intervalIntegral.integral_const, smul_eq_mul, mul_comm] at h
    exact h
  -- ∫ f ≤ M (b - a): ολοκληρώνουμε την f(x) ≤ M
  have h2 : ∫ x in a..b, f x ≤ f x₂ * (b - a) := by
    have h := intervalIntegral.integral_mono_on hab.le hint (intervalIntegrable_const (c := f x₂))
      (fun x hx => isMaxOn_iff.mp hmax x hx)
    rw [intervalIntegral.integral_const, smul_eq_mul, mul_comm] at h
    exact h
  -- διαιρούμε με b - a > 0: m ≤ μέση τιμή ≤ M
  have hba : 0 < b - a := sub_pos.mpr hab
  have h3 : f x₁ ≤ 1 / (b - a) * ∫ x in a..b, f x := by
    rw [one_div_mul_eq_div, le_div_iff₀ hba]
    exact h1
  have h4 : 1 / (b - a) * ∫ x in a..b, f x ≤ f x₂ := by
    rw [one_div_mul_eq_div, div_le_iff₀ hba]
    exact h2
  -- θεώρημα ενδιάμεσων τιμών στο διάστημα με άκρα x₁ και x₂
  have hsub : Set.uIcc x₁ x₂ ⊆ Set.Icc a b := Set.uIcc_subset_Icc hx₁ hx₂
  obtain ⟨ξ, hξ, hfξ⟩ := intermediate_value_uIcc (hf.mono hsub) (Set.mem_uIcc_of_le h3 h4)
  exact ⟨ξ, hsub hξ, hfξ⟩

theorem root_of_zero_integral (f : ℝ → ℝ) (a b : ℝ) (hab : a < b)
    (hf : ContinuousOn f (Set.Icc a b)) (h0 : ∫ x in a..b, f x = 0) :
    ∃ ξ ∈ Set.Icc a b, f ξ = 0 := by
  obtain ⟨ξ, hξ, hfξ⟩ := integral_mean_value f a b hab hf
  rw [h0, mul_zero] at hfξ
  exact ⟨ξ, hξ, hfξ⟩

end Κεφ14.MesiTimi
