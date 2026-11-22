-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 11 · Πολυώνυμο Taylor · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ11.PolyonymoTaylor
open scoped Nat

-- το πολυώνυμο Taylor της Mathlib: T_n(x) = Σ f^(k)(x₀)/k! · (x - x₀)^k
theorem taylor_formula (f : ℝ → ℝ) (n : ℕ) (s : Set ℝ) (x₀ x : ℝ) :
    taylorWithinEval f n s x₀ x =
      ∑ k ∈ Finset.range (n + 1), iteratedDerivWithin k f s x₀ / k ! * (x - x₀) ^ k := by
  rw [taylor_within_apply]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [smul_eq_mul]
  ring

theorem taylor_at_center (f : ℝ → ℝ) (n : ℕ) (s : Set ℝ) (x₀ : ℝ) :
    taylorWithinEval f n s x₀ x₀ = f x₀ :=
  taylorWithinEval_self f n s x₀

-- υπόλοιπο Lagrange: R_n(x) = f^(n+1)(ξ)/(n+1)! · (x - x₀)^(n+1), με ξ μεταξύ x₀ και x
theorem lagrange_remainder (f : ℝ → ℝ) (n : ℕ) (x₀ x : ℝ) (hx : x₀ ≠ x)
    (hf : ContDiffOn ℝ n f (Set.uIcc x₀ x))
    (hf' : DifferentiableOn ℝ (iteratedDerivWithin n f (Set.uIcc x₀ x)) (Set.uIoo x₀ x)) :
    ∃ ξ ∈ Set.uIoo x₀ x, f x - taylorWithinEval f n (Set.uIcc x₀ x) x₀ x =
      iteratedDerivWithin (n + 1) f (Set.uIcc x₀ x) ξ * (x - x₀) ^ (n + 1) / (n + 1)! :=
  taylor_mean_remainder_lagrange hx hf hf'

end Κεφ11.PolyonymoTaylor
