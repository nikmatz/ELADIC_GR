-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 17 · Εμβαδόν Επίπεδου Χωρίου · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (23/11/2026).

import Mathlib

namespace Κεφ17.EmvadonXwriou
open MeasureTheory

-- όταν f ≥ g: το εμβαδόν είναι το ολοκλήρωμα της f - g
theorem area_ordered (f g : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b)
    (hf : ContinuousOn f (Set.Icc a b)) (hg : ContinuousOn g (Set.Icc a b))
    (hgf : ∀ x ∈ Set.Icc a b, g x ≤ f x) :
    volume (regionBetween g f (Set.Icc a b)) = ENNReal.ofReal (∫ x in a..b, (f x - g x)) := by
  rw [Measure.volume_eq_prod, volume_regionBetween_eq_integral hg.integrableOn_Icc
    hf.integrableOn_Icc measurableSet_Icc hgf, intervalIntegral.integral_of_le hab,
    integral_Icc_eq_integral_Ioc]
  rfl

-- όταν οι καμπύλες αλλάζουν σχετική θέση: το εμβαδόν είναι το ολοκλήρωμα της |f - g|
theorem area_abs (f g : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b)
    (hf : ContinuousOn f (Set.Icc a b)) (hg : ContinuousOn g (Set.Icc a b)) :
    volume (regionBetween (fun x => min (f x) (g x)) (fun x => max (f x) (g x)) (Set.Icc a b)) =
      ENNReal.ofReal (∫ x in a..b, |f x - g x|) := by
  rw [Measure.volume_eq_prod, volume_regionBetween_eq_integral (hf.inf hg).integrableOn_Icc
    (hf.sup hg).integrableOn_Icc measurableSet_Icc (fun x _ => min_le_max),
    intervalIntegral.integral_of_le hab, integral_Icc_eq_integral_Ioc]
  simp only [Pi.sub_apply, max_sub_min_eq_abs']

end Κεφ17.EmvadonXwriou
