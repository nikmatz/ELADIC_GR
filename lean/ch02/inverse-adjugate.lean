-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 2 · Τύπος Αντιστρόφου μέσω Adjugate · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ2.AntistrofosProsartimenos
open Matrix

variable {n : ℕ}

theorem mul_adj (A : Matrix (Fin n) (Fin n) ℝ) :
    A * A.adjugate = A.det • (1 : Matrix (Fin n) (Fin n) ℝ) :=
  mul_adjugate A

theorem inv_eq (A : Matrix (Fin n) (Fin n) ℝ) (h : A.det ≠ 0) :
    A⁻¹ = (1 / A.det) • A.adjugate := by
  apply inv_eq_right_inv
  rw [Matrix.mul_smul, mul_adjugate, smul_smul, one_div, inv_mul_cancel₀ h, one_smul]

end Κεφ2.AntistrofosProsartimenos
