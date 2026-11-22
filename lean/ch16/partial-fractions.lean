-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 16 · Ολοκλήρωση με Μερικά Κλάσματα · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ16.MerikaKlasmata
open Polynomial

-- ύπαρξη: P = q · ∏ gᵢ^nᵢ + Σᵢ Σⱼ rᵢⱼ · gᵢ^j · ∏ (k ≠ i) gₖ^nₖ, με deg rᵢⱼ < deg gᵢ
theorem decomposition (P : ℝ[X]) {ι : Type*} [DecidableEq ι] (s : Finset ι) (g : ι → ℝ[X])
    (hg : ∀ i ∈ s, (g i).Monic) (hcop : Set.Pairwise s fun i j => IsCoprime (g i) (g j))
    (n : ι → ℕ) :
    ∃ (q : ℝ[X]) (r : (i : ι) → Fin (n i) → ℝ[X]),
      (∀ i ∈ s, ∀ j, (r i j).degree < (g i).degree) ∧
      P = q * ∏ i ∈ s, g i ^ n i + ∑ i ∈ s, ∑ j, r i j * g i ^ j.val * ∏ k ∈ s.erase i, g k ^ n k :=
  eq_quo_mul_prod_pow_add_sum_rem_mul_prod_pow P hg hcop n

-- μοναδικότητα: δύο τέτοιες αναλύσεις έχουν το ίδιο q και τα ίδια rᵢⱼ
theorem uniqueness {ι : Type*} [DecidableEq ι] (s : Finset ι) (g : ι → ℝ[X])
    (hg : ∀ i ∈ s, (g i).Monic) (hcop : Set.Pairwise s fun i j => IsCoprime (g i) (g j))
    (n : ι → ℕ) (q₁ q₂ : ℝ[X]) (r₁ r₂ : (i : ι) → Fin (n i) → ℝ[X])
    (hr₁ : ∀ i ∈ s, ∀ j, (r₁ i j).degree < (g i).degree)
    (hr₂ : ∀ i ∈ s, ∀ j, (r₂ i j).degree < (g i).degree)
    (h : q₁ * ∏ i ∈ s, g i ^ n i + ∑ i ∈ s, ∑ j, r₁ i j * g i ^ j.val * ∏ k ∈ s.erase i, g k ^ n k =
      q₂ * ∏ i ∈ s, g i ^ n i + ∑ i ∈ s, ∑ j, r₂ i j * g i ^ j.val * ∏ k ∈ s.erase i, g k ^ n k) :
    q₁ = q₂ ∧ ∀ i ∈ s, r₁ i = r₂ i :=
  quo_mul_prod_pow_add_sum_rem_mul_prod_pow_unique hg hcop hr₁ hr₂ h

-- παράδειγμα: 1/(x² - 1) = (1/2)/(x - 1) - (1/2)/(x + 1)
theorem example_decomposition (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
    1 / (x ^ 2 - 1) = 1 / 2 / (x - 1) - 1 / 2 / (x + 1) := by
  have h1' : x - 1 ≠ 0 := sub_ne_zero.mpr h1
  have h2' : x + 1 ≠ 0 := fun h => h2 (by linarith)
  have h3 : x ^ 2 - 1 ≠ 0 := by
    rw [show x ^ 2 - 1 = (x - 1) * (x + 1) by ring]
    exact mul_ne_zero h1' h2'
  field_simp
  ring

-- και το ολοκλήρωμα: ∫ dx/(x² - 1) = (1/2) ln|x - 1| - (1/2) ln|x + 1| + C
theorem example_integral (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
    HasDerivAt (fun x => 1 / 2 * Real.log |x - 1| - 1 / 2 * Real.log |x + 1|)
      (1 / (x ^ 2 - 1)) x := by
  simp only [Real.log_abs]
  have ha : HasDerivAt (fun x => Real.log (x - 1)) (1 / (x - 1)) x :=
    ((hasDerivAt_id' x).sub_const 1).log (sub_ne_zero.mpr h1)
  have hb : HasDerivAt (fun x => Real.log (x + 1)) (1 / (x + 1)) x :=
    ((hasDerivAt_id' x).add_const 1).log (fun h => h2 (by linarith))
  have h : HasDerivAt (fun x => 1 / 2 * Real.log (x - 1) - 1 / 2 * Real.log (x + 1))
      (1 / 2 * (1 / (x - 1)) - 1 / 2 * (1 / (x + 1))) x :=
    (ha.const_mul (1 / 2)).sub (hb.const_mul (1 / 2))
  convert h using 1
  rw [example_decomposition x h1 h2]
  ring

end Κεφ16.MerikaKlasmata
