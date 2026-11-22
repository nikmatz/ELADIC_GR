-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 1 · Βασικές Ιδιότητες Γραμμικού Χώρου · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

/-!
# Βασικές Ιδιότητες Γραμμικού Χώρου (Κεφάλαιο 1)

Ορίζουμε τον γραμμικό χώρο όπως στο βιβλίο: ένα σύνολο «V» με πρόσθεση,
μηδενικό διάνυσμα, αντίθετα και βαθμωτό πολλαπλασιασμό με πραγματικούς
αριθμούς, που ικανοποιούν οκτώ αξιώματα. Κάθε ιδιότητα παρακάτω
αποδεικνύεται μόνο από αυτά τα αξιώματα.
(Το «λ» είναι δεσμευμένη λέξη του Lean, γι' αυτό τους αριθμούς τους λέμε «κ», «μ».)
-/

/-- Γραμμικός (διανυσματικός) χώρος πάνω στο ℝ, με τα οκτώ αξιώματα του βιβλίου. -/
class VectorSpace (V : Type) extends Add V, Zero V, Neg V, SMul ℝ V where
  /-- u + v = v + u -/
  add_comm : ∀ u v : V, u + v = v + u
  /-- (u + v) + w = u + (v + w) -/
  add_assoc : ∀ u v w : V, u + v + w = u + (v + w)
  /-- u + 0 = u -/
  add_zero : ∀ u : V, u + 0 = u
  /-- u + (−u) = 0 -/
  add_neg : ∀ u : V, u + -u = 0
  /-- κ·(u + v) = κ·u + κ·v -/
  smul_add : ∀ (κ : ℝ) (u v : V), κ • (u + v) = κ • u + κ • v
  /-- (κ + μ)·u = κ·u + μ·u -/
  add_smul : ∀ (κ μ : ℝ) (u : V), (κ + μ) • u = κ • u + μ • u
  /-- (κμ)·u = κ·(μ·u) -/
  mul_smul : ∀ (κ μ : ℝ) (u : V), (κ * μ) • u = κ • (μ • u)
  /-- 1·u = u -/
  one_smul : ∀ u : V, (1 : ℝ) • u = u

namespace Κεφ1

variable {V : Type} [VectorSpace V]

/-- (1) Το μηδενικό διάνυσμα είναι μοναδικό. -/
theorem zero_unique (z : V) (hz : ∀ u : V, u + z = u) : z = 0 := by
  calc z = z + 0 := (VectorSpace.add_zero z).symm
    _ = 0 + z := VectorSpace.add_comm z 0
    _ = 0 := hz 0

/-- (2) Το αντίθετο κάθε διανύσματος είναι μοναδικό. -/
theorem neg_unique {u w : V} (h : u + w = 0) : w = -u := by
  calc w = w + 0 := (VectorSpace.add_zero w).symm
    _ = w + (u + -u) := by rw [VectorSpace.add_neg]
    _ = w + u + -u := (VectorSpace.add_assoc w u (-u)).symm
    _ = u + w + -u := by rw [VectorSpace.add_comm w u]
    _ = 0 + -u := by rw [h]
    _ = -u + 0 := VectorSpace.add_comm 0 (-u)
    _ = -u := VectorSpace.add_zero (-u)

/-- Λήμμα: αν x + x = x, τότε x = 0 (προσθέτουμε το −x και στα δύο μέλη). -/
theorem eq_zero_of_add_self {x : V} (h : x + x = x) : x = 0 := by
  calc x = x + 0 := (VectorSpace.add_zero x).symm
    _ = x + (x + -x) := by rw [VectorSpace.add_neg]
    _ = x + x + -x := (VectorSpace.add_assoc x x (-x)).symm
    _ = x + -x := by rw [h]
    _ = 0 := VectorSpace.add_neg x

/-- (3) 0·u = 0 -/
theorem zero_smul' (u : V) : (0 : ℝ) • u = 0 := by
  apply eq_zero_of_add_self
  rw [← VectorSpace.add_smul]
  rw [add_zero]

/-- (4) κ·0 = 0 -/
theorem smul_zero' (κ : ℝ) : κ • (0 : V) = 0 := by
  apply eq_zero_of_add_self
  rw [← VectorSpace.smul_add]
  rw [VectorSpace.add_zero]

/-- (5) (−1)·u = −u -/
theorem neg_one_smul' (u : V) : (-1 : ℝ) • u = -u := by
  apply neg_unique
  calc u + (-1 : ℝ) • u = (1 : ℝ) • u + (-1 : ℝ) • u := by rw [VectorSpace.one_smul]
    _ = ((1 : ℝ) + -1) • u := (VectorSpace.add_smul 1 (-1) u).symm
    _ = (0 : ℝ) • u := by norm_num
    _ = 0 := zero_smul' u

/-- (6) Αν κ·u = 0, τότε κ = 0 ή u = 0. -/
theorem eq_zero_or_eq_zero_of_smul_eq_zero {κ : ℝ} {u : V} (h : κ • u = 0) :
    κ = 0 ∨ u = 0 := by
  by_cases hκ : κ = 0
  · left
    exact hκ
  · right
    calc u = (1 : ℝ) • u := (VectorSpace.one_smul u).symm
      _ = (κ⁻¹ * κ) • u := by rw [inv_mul_cancel₀ hκ]
      _ = κ⁻¹ • (κ • u) := VectorSpace.mul_smul κ⁻¹ κ u
      _ = κ⁻¹ • (0 : V) := by rw [h]
      _ = 0 := smul_zero' κ⁻¹

end Κεφ1

/-! ## Οι ίδιες ιδιότητες είναι έτοιμες στη Mathlib (εκεί ο γραμμικός χώρος λέγεται «Module») -/
section InMathlib
variable {W : Type} [AddCommGroup W] [Module ℝ W]

example (u : W) : (0 : ℝ) • u = 0 := zero_smul ℝ u
example (κ : ℝ) : κ • (0 : W) = 0 := smul_zero κ
example (u : W) : (-1 : ℝ) • u = -u := neg_one_smul ℝ u
example (κ : ℝ) (u : W) : κ • u = 0 ↔ κ = 0 ∨ u = 0 := smul_eq_zero
end InMathlib
