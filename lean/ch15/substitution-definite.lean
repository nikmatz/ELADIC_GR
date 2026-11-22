-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 15 · Αντικατάσταση σε Ορισμένο Ολοκλήρωμα (Αλλαγή Ορίων) · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ15.AntikatastasiOrismeno
open MeasureTheory

theorem subst_definite (f g g' : ℝ → ℝ) (a b : ℝ) (hf : Continuous f)
    (hg : ∀ x ∈ Set.uIcc a b, HasDerivAt g (g' x) x) (hg' : ContinuousOn g' (Set.uIcc a b)) :
    ∫ x in a..b, f (g x) * g' x = ∫ u in g a..g b, f u := by
  -- η F(u) = ∫ f(t) dt από g(a) έως u έχει παράγωγο f(u)
  have hF : ∀ u, HasDerivAt (fun u => ∫ t in g a..u, f t) (f u) u :=
    fun u => (hf.integral_hasStrictDerivAt (g a) u).hasDerivAt
  -- κανόνας αλυσίδας: (F(g(x)))' = f(g(x)) g'(x)
  have hcomp : ∀ x ∈ Set.uIcc a b,
      HasDerivAt (fun x => ∫ t in g a..g x, f t) (f (g x) * g' x) x :=
    fun x hx => (hF (g x)).comp x (hg x hx)
  -- η f(g(x)) g'(x) είναι συνεχής, άρα ολοκληρώσιμη
  have hgc : ContinuousOn g (Set.uIcc a b) := fun x hx => (hg x hx).continuousAt.continuousWithinAt
  have hint : IntervalIntegrable (fun x => f (g x) * g' x) volume a b :=
    ((hf.comp_continuousOn hgc).mul hg').intervalIntegrable
  -- θεμελιώδες θεώρημα: το ολοκλήρωμα είναι F(g(b)) - F(g(a))
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hcomp hint]
  -- F(g(a)) = 0, γιατί τα άκρα είναι ίσα
  rw [intervalIntegral.integral_same, sub_zero]

theorem shift (f : ℝ → ℝ) (a b c : ℝ) (hf : Continuous f) :
    ∫ x in a..b, f (x + c) = ∫ u in a + c..b + c, f u := by
  have h := subst_definite f (fun x => x + c) (fun _ => 1) a b hf
    (fun x _ => (hasDerivAt_id' x).add_const c) continuousOn_const
  simp only [mul_one] at h
  exact h

end Κεφ15.AntikatastasiOrismeno
