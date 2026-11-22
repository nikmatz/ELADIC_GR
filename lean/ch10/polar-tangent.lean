-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 10 · Κλίση Εφαπτομένης Πολικής Καμπύλης · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ10.Poliki
open Filter Topology

theorem x_deriv (r : ℝ → ℝ) (θ r' : ℝ) (hr : HasDerivAt r r' θ) :
    HasDerivAt (fun t => r t * Real.cos t) (r' * Real.cos θ - r θ * Real.sin θ) θ := by
  have h := hr.mul (Real.hasDerivAt_cos θ)
  rw [show r' * Real.cos θ - r θ * Real.sin θ = r' * Real.cos θ + r θ * -Real.sin θ by ring]
  exact h

theorem y_deriv (r : ℝ → ℝ) (θ r' : ℝ) (hr : HasDerivAt r r' θ) :
    HasDerivAt (fun t => r t * Real.sin t) (r' * Real.sin θ + r θ * Real.cos θ) θ :=
  hr.mul (Real.hasDerivAt_sin θ)

theorem polar_slope (r F : ℝ → ℝ) (θ₀ r' F' : ℝ) (hr : HasDerivAt r r' θ₀)
    (hF : HasDerivAt F F' (r θ₀ * Real.cos θ₀))
    (hcurve : ∀ᶠ t in 𝓝 θ₀, r t * Real.sin t = F (r t * Real.cos t))
    (hden : r' * Real.cos θ₀ - r θ₀ * Real.sin θ₀ ≠ 0) :
    F' = (r' * Real.sin θ₀ + r θ₀ * Real.cos θ₀) / (r' * Real.cos θ₀ - r θ₀ * Real.sin θ₀) := by
  have hx := x_deriv r θ₀ r' hr
  have hy := y_deriv r θ₀ r' hr
  have h1 := hF.comp θ₀ hx
  have h2 : HasDerivAt (fun t => r t * Real.sin t)
      (F' * (r' * Real.cos θ₀ - r θ₀ * Real.sin θ₀)) θ₀ := h1.congr_of_eventuallyEq hcurve
  have h3 := hy.unique h2
  rw [h3, mul_div_cancel_right₀ _ hden]

-- οριζόντια εφαπτομένη
theorem polar_horizontal (r F : ℝ → ℝ) (θ₀ r' F' : ℝ) (hr : HasDerivAt r r' θ₀)
    (hF : HasDerivAt F F' (r θ₀ * Real.cos θ₀))
    (hcurve : ∀ᶠ t in 𝓝 θ₀, r t * Real.sin t = F (r t * Real.cos t))
    (hden : r' * Real.cos θ₀ - r θ₀ * Real.sin θ₀ ≠ 0)
    (hnum : r' * Real.sin θ₀ + r θ₀ * Real.cos θ₀ = 0) : F' = 0 := by
  rw [polar_slope r F θ₀ r' F' hr hF hcurve hden, hnum, zero_div]

-- κατακόρυφη εφαπτομένη: η καμπύλη γράφεται x = G(y) και dx/dy = 0
theorem polar_vertical (r G : ℝ → ℝ) (θ₀ r' G' : ℝ) (hr : HasDerivAt r r' θ₀)
    (hG : HasDerivAt G G' (r θ₀ * Real.sin θ₀))
    (hcurve : ∀ᶠ t in 𝓝 θ₀, r t * Real.cos t = G (r t * Real.sin t))
    (hnum : r' * Real.sin θ₀ + r θ₀ * Real.cos θ₀ ≠ 0)
    (hden : r' * Real.cos θ₀ - r θ₀ * Real.sin θ₀ = 0) : G' = 0 := by
  have hx := x_deriv r θ₀ r' hr
  have hy := y_deriv r θ₀ r' hr
  have h1 := hG.comp_of_eq θ₀ hy rfl
  have h2 : HasDerivAt (fun t => r t * Real.cos t)
      (G' * (r' * Real.sin θ₀ + r θ₀ * Real.cos θ₀)) θ₀ := h1.congr_of_eventuallyEq hcurve
  have h3 := hx.unique h2
  rw [hden] at h3
  exact (mul_eq_zero.mp h3.symm).resolve_right hnum

end Κεφ10.Poliki
