-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · Δομή Σώματος του ℂ · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.SomaC
open Complex ComplexConjugate

theorem comm_add (z w : ℂ) : z + w = w + z := add_comm z w

theorem comm_mul (z w : ℂ) : z * w = w * z := mul_comm z w

theorem assoc_add (z w u : ℂ) : z + w + u = z + (w + u) := add_assoc z w u

theorem assoc_mul (z w u : ℂ) : z * w * u = z * (w * u) := mul_assoc z w u

theorem distrib (z w u : ℂ) : z * (w + u) = z * w + z * u := mul_add z w u

theorem zero_neutral (z : ℂ) : z + 0 = z := add_zero z

theorem one_neutral (z : ℂ) : z * 1 = z := mul_one z

theorem opposite (z : ℂ) : z + -z = 0 := add_neg_cancel z

theorem inverse (z : ℂ) (hz : z ≠ 0) : z * z⁻¹ = 1 := mul_inv_cancel₀ hz

theorem inverse_formula (z : ℂ) : z⁻¹ = conj z * ((normSq z)⁻¹ : ℝ) := inv_def z

theorem is_field : IsField ℂ := Field.toIsField ℂ

end Κεφ0.SomaC
