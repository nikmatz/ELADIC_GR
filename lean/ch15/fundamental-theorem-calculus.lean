-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 15 · Θεμελιώδες Θεώρημα Λογισμού · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ15.ThemeliodesThewrima
open MeasureTheory

-- Μέρος Α: η G(x) = ∫ f(t) dt από a έως x έχει παράγωγο f(x) σε κάθε x ∈ (a, b)
theorem ftc_part_A (f : ℝ → ℝ) (a b : ℝ) (hf : ContinuousOn f (Set.Icc a b))
    (x : ℝ) (hx : x ∈ Set.Ioo a b) :
    HasDerivAt (fun u => ∫ t in a..u, f t) (f x) x := by
  have hint : IntervalIntegrable f volume a x :=
    (hf.mono (Set.Icc_subset_Icc le_rfl hx.2.le)).intervalIntegrable_of_Icc hx.1.le
  have hmeas : StronglyMeasurableAtFilter f (nhds x) volume :=
    (hf.mono Set.Ioo_subset_Icc_self).stronglyMeasurableAtFilter isOpen_Ioo x hx
  exact intervalIntegral.integral_hasDerivAt_right hint hmeas
    (hf.continuousAt (Icc_mem_nhds hx.1 hx.2))

-- Μέρος Β (Newton–Leibniz): αν F' = f, τότε το ολοκλήρωμα της f από a έως b είναι F(b) - F(a)
theorem ftc_part_B (f F : ℝ → ℝ) (a b : ℝ) (hab : a ≤ b) (hF : ContinuousOn F (Set.Icc a b))
    (hFf : ∀ x ∈ Set.Ioo a b, HasDerivAt F (f x) x) (hf : ContinuousOn f (Set.Icc a b)) :
    ∫ x in a..b, f x = F b - F a :=
  intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le hab hF hFf (hf.intervalIntegrable_of_Icc hab)

end Κεφ15.ThemeliodesThewrima
