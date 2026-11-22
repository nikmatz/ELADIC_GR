-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 19 · Κριτήριο Ρίζας (Root Test) · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ19.KritirioRizas
open Filter Topology

-- για μεγάλα n η ⁿ√|a n| είναι ≤ q, με q < 1 (αυτό σημαίνει limsup ⁿ√|a n| < 1)
theorem root_test (a : ℕ → ℝ) (q : ℝ) (hq : q < 1)
    (h : ∀ᶠ n in atTop, |a n| ^ ((n : ℝ)⁻¹) ≤ q) : Summable a := by
  -- για μεγάλα n: |a n| ≤ qⁿ
  have h1 : ∀ᶠ n in atTop, |a n| ≤ q ^ n := by
    filter_upwards [h, eventually_ge_atTop 1] with n hn hn1
    have hn0 : n ≠ 0 := by omega
    calc |a n| = (|a n| ^ ((n : ℝ)⁻¹)) ^ n := (Real.rpow_inv_natCast_pow (abs_nonneg _) hn0).symm
      _ ≤ q ^ n := pow_le_pow_left₀ (Real.rpow_nonneg (abs_nonneg _) _) hn n
  -- το q είναι μη αρνητικό
  have hq0 : 0 ≤ q := by
    obtain ⟨n, hn⟩ := h.exists
    exact (Real.rpow_nonneg (abs_nonneg _) _).trans hn
  -- η γεωμετρική σειρά Σ qⁿ συγκλίνει, γιατί 0 ≤ q < 1
  have hgeom : Summable (fun n => q ^ n) := summable_geometric_of_lt_one hq0 hq
  -- σύγκριση με τη γεωμετρική σειρά
  exact hgeom.of_norm_bounded_eventually_nat (by simpa [Real.norm_eq_abs] using h1)

theorem root_test_limit (a : ℕ → ℝ) (L : ℝ) (hL : L < 1)
    (h : Tendsto (fun n => |a n| ^ ((n : ℝ)⁻¹)) atTop (𝓝 L)) : Summable a :=
  root_test a ((L + 1) / 2) (by linarith) (h.eventually (ge_mem_nhds (by linarith)))

-- αν συχνά ⁿ√|a n| > 1 (limsup > 1), τα μερικά αθροίσματα δεν συγκλίνουν
theorem root_test_diverges (a : ℕ → ℝ) (h : ∃ᶠ n in atTop, 1 < |a n| ^ ((n : ℝ)⁻¹)) :
    ¬ ∃ S, Tendsto (fun N => ∑ n ∈ Finset.range N, a n) atTop (𝓝 S) := by
  rintro ⟨S, hS⟩
  have h0 : Tendsto a atTop (𝓝 0) := by
    have h1 := (hS.comp (tendsto_add_atTop_nat 1)).sub hS
    simp only [Function.comp_def, Finset.sum_range_succ, add_sub_cancel_left, sub_self] at h1
    exact h1
  have h2 : ∀ᶠ n in atTop, |a n| < 1 := by
    have h3 := h0.abs.eventually (gt_mem_nhds (show |(0 : ℝ)| < 1 by norm_num))
    exact h3
  obtain ⟨n, hn1, hn2⟩ := (h.and_eventually h2).exists
  have h4 := Real.rpow_le_one (abs_nonneg (a n)) hn2.le (inv_nonneg.mpr (Nat.cast_nonneg n))
  linarith

end Κεφ19.KritirioRizas
