-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 20 · Σειρά Taylor και Maclaurin · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ20.SeiraTaylor
open Filter Topology
open scoped Nat

-- T_n(x) = Σ από k = 0 έως n των f^(k)(x₀)/k! · (x - x₀)^k, με παραγώγους μέσα στο σύνολο s
theorem taylor_partial_sum (f : ℝ → ℝ) (n : ℕ) (s : Set ℝ) (x₀ x : ℝ) :
    taylorWithinEval f n s x₀ x =
      ∑ k ∈ Finset.range (n + 1), iteratedDerivWithin k f s x₀ / k ! * (x - x₀) ^ k := by
  rw [taylor_within_apply]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [smul_eq_mul]
  ring

-- η σειρά Taylor έχει άθροισμα f(x) αν και μόνο αν R_n(x) = f(x) - T_n(x) → 0
theorem taylor_series_iff (f : ℝ → ℝ) (s : Set ℝ) (x₀ x : ℝ) :
    Tendsto (fun n => taylorWithinEval f n s x₀ x) atTop (𝓝 (f x)) ↔
      Tendsto (fun n => f x - taylorWithinEval f n s x₀ x) atTop (𝓝 0) := by
  constructor
  · -- (⇒) αν T_n(x) → f(x), τότε f(x) - T_n(x) → f(x) - f(x) = 0
    intro h
    have h1 := (tendsto_const_nhds (x := f x)).sub h
    rw [sub_self] at h1
    exact h1
  · -- (⇐) αν R_n(x) → 0, τότε T_n(x) = f(x) - R_n(x) → f(x) - 0 = f(x)
    intro h
    have h1 := (tendsto_const_nhds (x := f x)).sub h
    rw [sub_zero] at h1
    have h2 : (fun n => f x - (f x - taylorWithinEval f n s x₀ x)) =
        fun n => taylorWithinEval f n s x₀ x := by
      funext n
      ring
    rw [h2] at h1
    exact h1

-- για x₀ = 0: η σειρά Maclaurin
theorem maclaurin_iff (f : ℝ → ℝ) (s : Set ℝ) (x : ℝ) :
    Tendsto (fun n => taylorWithinEval f n s 0 x) atTop (𝓝 (f x)) ↔
      Tendsto (fun n => f x - taylorWithinEval f n s 0 x) atTop (𝓝 0) :=
  taylor_series_iff f s 0 x

-- υπόλοιπο Lagrange: R_n(x) = f^(n+1)(ξ)/(n+1)! · (x - x₀)^(n+1), με ξ μεταξύ x₀ και x
theorem lagrange_remainder (f : ℝ → ℝ) (n : ℕ) (x₀ x : ℝ) (hx : x₀ ≠ x)
    (hf : ContDiffOn ℝ n f (Set.uIcc x₀ x))
    (hf' : DifferentiableOn ℝ (iteratedDerivWithin n f (Set.uIcc x₀ x)) (Set.uIoo x₀ x)) :
    ∃ ξ ∈ Set.uIoo x₀ x, f x - taylorWithinEval f n (Set.uIcc x₀ x) x₀ x =
      iteratedDerivWithin (n + 1) f (Set.uIcc x₀ x) ξ * (x - x₀) ^ (n + 1) / (n + 1)! :=
  taylor_mean_remainder_lagrange hx hf hf'

-- στο κέντρο x₀ η σειρά Taylor συγκλίνει πάντα στο f(x₀)
theorem taylor_at_center (f : ℝ → ℝ) (s : Set ℝ) (x₀ : ℝ) :
    Tendsto (fun n => taylorWithinEval f n s x₀ x₀) atTop (𝓝 (f x₀)) := by
  simp only [taylorWithinEval_self]
  exact tendsto_const_nhds

end Κεφ20.SeiraTaylor
