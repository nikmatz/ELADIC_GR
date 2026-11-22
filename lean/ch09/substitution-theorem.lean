-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 9 · Θεώρημα Αντικατάστασης · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ9.Antikatastasi
open Filter Topology

-- ο ορισμός του βιβλίου: lim_{x→x₀} f(x) = L
def HasLim (f : ℝ → ℝ) (x₀ L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| → |x - x₀| < δ → |f x - L| < ε

-- η συνέχεια της f στο ℓ, επίσης με ε και δ
def ContAt (f : ℝ → ℝ) (ℓ : ℝ) : Prop :=
  ∀ ε > 0, ∃ η > 0, ∀ u, |u - ℓ| < η → |f u - f ℓ| < ε

theorem lim_comp (f g : ℝ → ℝ) (x₀ ℓ : ℝ) (hg : HasLim g x₀ ℓ) (hf : ContAt f ℓ) :
    HasLim (f ∘ g) x₀ (f ℓ) := by
  intro ε hε
  obtain ⟨η, hη, h₁⟩ := hf ε hε
  obtain ⟨δ, hδ, h₂⟩ := hg η hη
  refine ⟨δ, hδ, ?_⟩
  intro x hx hxδ
  have hgx : |g x - ℓ| < η := h₂ x hx hxδ
  exact h₁ (g x) hgx

theorem cont_id (ℓ : ℝ) : ContAt (fun u => u) ℓ := by
  intro ε hε
  refine ⟨ε, hε, ?_⟩
  intro u hu
  exact hu

theorem lim_comp_mathlib (f g : ℝ → ℝ) (x₀ ℓ : ℝ) (hg : Tendsto g (𝓝[≠] x₀) (𝓝 ℓ))
    (hf : ContinuousAt f ℓ) : Tendsto (f ∘ g) (𝓝[≠] x₀) (𝓝 (f ℓ)) :=
  hf.tendsto.comp hg

end Κεφ9.Antikatastasi
