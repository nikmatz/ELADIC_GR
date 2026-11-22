-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · Τρεις Περιπτώσεις — Γεωμετρική Ερμηνεία · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.Trichotomia
open Matrix

variable {m n : ℕ}

theorem difference (A : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ) (x y : Fin n → ℝ)
    (hx : A *ᵥ x = b) (hy : A *ᵥ y = b) : A *ᵥ (y - x) = 0 := by
  rw [mulVec_sub, hx, hy, sub_self]

theorem line_of_solutions (A : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ) (x y : Fin n → ℝ)
    (hx : A *ᵥ x = b) (hy : A *ᵥ y = b) (t : ℝ) : A *ᵥ (x + t • (y - x)) = b := by
  rw [mulVec_add]
  rw [mulVec_smul]
  rw [difference A b x y hx hy]
  rw [smul_zero, add_zero, hx]

theorem trichotomy (A : Matrix (Fin m) (Fin n) ℝ) (b : Fin m → ℝ) :
    {x | A *ᵥ x = b} = ∅ ∨ (∃ x, {x | A *ᵥ x = b} = {x}) ∨ {x | A *ᵥ x = b}.Infinite := by
  by_cases h0 : {x | A *ᵥ x = b}.Nonempty
  · right
    obtain ⟨x, hx⟩ := h0
    by_cases h1 : ∀ y, A *ᵥ y = b → y = x
    · left
      refine ⟨x, ?_⟩
      ext y
      constructor
      · intro hy
        exact h1 y hy
      · intro hy
        rw [Set.mem_singleton_iff] at hy
        rw [hy]
        exact hx
    · right
      push Not at h1
      obtain ⟨y, hy, hne⟩ := h1
      have hd : y - x ≠ 0 := sub_ne_zero.mpr hne
      apply Set.infinite_of_injective_forall_mem (f := fun t : ℝ => x + t • (y - x))
      · intro s t hst
        dsimp only at hst
        have h' : s • (y - x) = t • (y - x) := add_left_cancel hst
        exact smul_left_injective ℝ hd h'
      · intro t
        exact line_of_solutions A b x y hx hy t
  · left
    exact Set.not_nonempty_iff_eq_empty.mp h0

theorem exclusive (S : Set (Fin n → ℝ)) :
    ¬ (S = ∅ ∧ ∃ x, S = {x}) ∧ ¬ (S = ∅ ∧ S.Infinite) ∧ ¬ ((∃ x, S = {x}) ∧ S.Infinite) := by
  refine ⟨?_, ?_, ?_⟩
  · rintro ⟨h₁, x, h₂⟩
    rw [h₁] at h₂
    exact (Set.singleton_nonempty x).ne_empty h₂.symm
  · rintro ⟨h₁, h₂⟩
    rw [h₁] at h₂
    exact h₂ Set.finite_empty
  · rintro ⟨⟨x, h₁⟩, h₂⟩
    rw [h₁] at h₂
    exact h₂ (Set.finite_singleton x)

end Κεφ0.Trichotomia
