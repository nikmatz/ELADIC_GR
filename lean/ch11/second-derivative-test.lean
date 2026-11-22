-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 11 · Κριτήριο 2ης Παραγώγου για Ακρότατα · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ11.KritirioDeuterisParagogou

theorem second_deriv_min (f : ℝ → ℝ) (x₀ : ℝ) (hc : ContinuousAt f x₀) (hd : deriv f x₀ = 0)
    (hpos : 0 < deriv (deriv f) x₀) : IsLocalMin f x₀ :=
  isLocalMin_of_deriv_deriv_pos hpos hd hc

theorem second_deriv_max (f : ℝ → ℝ) (x₀ : ℝ) (hc : ContinuousAt f x₀) (hd : deriv f x₀ = 0)
    (hneg : deriv (deriv f) x₀ < 0) : IsLocalMax f x₀ :=
  isLocalMax_of_deriv_deriv_neg hneg hd hc

-- όταν f''(x₀) = 0 το κριτήριο δεν αποφαίνεται: η x⁴ έχει ελάχιστο στο 0, η x³ κανένα ακρότατο
theorem quartic : deriv (fun x : ℝ => x ^ 4) 0 = 0 ∧ deriv (deriv (fun x : ℝ => x ^ 4)) 0 = 0 ∧
    IsLocalMin (fun x : ℝ => x ^ 4) 0 := by
  have h1 : deriv (fun x : ℝ => x ^ 4) = fun x => 4 * x ^ 3 := by
    funext x
    rw [deriv_pow_field]
    norm_num
  refine ⟨by rw [h1]; norm_num, ?_, ?_⟩
  · rw [h1, deriv_const_mul _ (differentiableAt_pow 3), deriv_pow_field]
    norm_num
  · exact Filter.Eventually.of_forall fun x => by
      show (0 : ℝ) ^ 4 ≤ x ^ 4
      rw [zero_pow (by norm_num)]
      positivity

theorem cubic : deriv (fun x : ℝ => x ^ 3) 0 = 0 ∧ deriv (deriv (fun x : ℝ => x ^ 3)) 0 = 0 ∧
    ¬ IsLocalExtr (fun x : ℝ => x ^ 3) 0 := by
  have h1 : deriv (fun x : ℝ => x ^ 3) = fun x => 3 * x ^ 2 := by
    funext x
    rw [deriv_pow_field]
    norm_num
  refine ⟨by rw [h1]; norm_num, ?_, ?_⟩
  · rw [h1, deriv_const_mul _ (differentiableAt_pow 2), deriv_pow_field]
    norm_num
  · have hpt : ∀ ε : ℝ, 0 < ε → dist (ε / 2) 0 < ε ∧ dist (-(ε / 2)) 0 < ε := by
      intro ε hε
      rw [Real.dist_eq, Real.dist_eq, sub_zero, sub_zero, abs_neg, abs_of_pos (half_pos hε)]
      constructor <;> linarith
    rintro (h | h)
    · obtain ⟨ε, hε, hball⟩ := Metric.eventually_nhds_iff.mp h
      have h2 : (0 : ℝ) ^ 3 ≤ (-(ε / 2)) ^ 3 := hball (hpt ε hε).2
      have h3 : 0 < (ε / 2) ^ 3 := pow_pos (half_pos hε) 3
      linarith
    · obtain ⟨ε, hε, hball⟩ := Metric.eventually_nhds_iff.mp h
      have h2 : (ε / 2) ^ 3 ≤ (0 : ℝ) ^ 3 := hball (hpt ε hε).1
      have h3 : 0 < (ε / 2) ^ 3 := pow_pos (half_pos hε) 3
      linarith

end Κεφ11.KritirioDeuterisParagogou
