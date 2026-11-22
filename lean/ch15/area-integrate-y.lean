-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 15 · Εμβαδόν με Ολοκλήρωση ως προς $y$ · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ15.EmvadonWsProsY
open MeasureTheory

-- το χωρίο: τα σημεία (x, y) με y ∈ [c, d] και q(y) < x < p(y)
theorem area_dy (p q : ℝ → ℝ) (c d : ℝ) (hcd : c ≤ d)
    (hp : ContinuousOn p (Set.Icc c d)) (hq : ContinuousOn q (Set.Icc c d))
    (hqp : ∀ y ∈ Set.Icc c d, q y ≤ p y) :
    volume {z : ℝ × ℝ | z.2 ∈ Set.Icc c d ∧ z.1 ∈ Set.Ioo (q z.2) (p z.2)} =
      ENNReal.ofReal (∫ y in c..d, (p y - q y)) := by
  -- η εναλλαγή των αξόνων (x, y) ↦ (y, x) διατηρεί το εμβαδόν
  have hswap : MeasurePreserving (MeasurableEquiv.prodComm : ℝ × ℝ ≃ᵐ ℝ × ℝ)
      (volume.prod volume) (volume.prod volume) := Measure.measurePreserving_swap
  -- μετά την εναλλαγή, το χωρίο είναι το χωρίο ανάμεσα στις γραφικές των q και p
  have hset : {z : ℝ × ℝ | z.2 ∈ Set.Icc c d ∧ z.1 ∈ Set.Ioo (q z.2) (p z.2)} =
      (MeasurableEquiv.prodComm : ℝ × ℝ ≃ᵐ ℝ × ℝ) ⁻¹' regionBetween q p (Set.Icc c d) := rfl
  rw [Measure.volume_eq_prod, hset, hswap.measure_preimage_equiv,
    volume_regionBetween_eq_integral hq.integrableOn_Icc hp.integrableOn_Icc measurableSet_Icc hqp,
    intervalIntegral.integral_of_le hcd, integral_Icc_eq_integral_Ioc]
  rfl

end Κεφ15.EmvadonWsProsY
