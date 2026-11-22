-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 10 · Κανόνας Αλυσίδας · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ10.KanonasAlysidas

theorem chain_rule (f g : ℝ → ℝ) (x f' g' : ℝ) (hg : HasDerivAt g g' x)
    (hf : HasDerivAt f f' (g x)) : HasDerivAt (fun t => f (g t)) (f' * g') x :=
  hf.comp x hg

theorem chain_rule_deriv (f g : ℝ → ℝ) (x : ℝ) (hg : DifferentiableAt ℝ g x)
    (hf : DifferentiableAt ℝ f (g x)) :
    deriv (fun t => f (g t)) x = deriv f (g x) * deriv g x :=
  (hf.hasDerivAt.comp x hg.hasDerivAt).deriv

end Κεφ10.KanonasAlysidas
