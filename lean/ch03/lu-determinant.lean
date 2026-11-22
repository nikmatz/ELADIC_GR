-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 3 · Σύνδεση LU — Ορίζουσα · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ3.OrizousaLU
open Matrix

variable {n : ℕ}

theorem det_L (L : Matrix (Fin n) (Fin n) ℝ) (hL : ∀ i j, i < j → L i j = 0)
    (h1 : ∀ i, L i i = 1) : L.det = 1 := by
  rw [det_of_isLowerTriangular L (fun i j hij => hL i j hij)]
  simp [h1]

theorem det_LU (L U : Matrix (Fin n) (Fin n) ℝ) (hL : ∀ i j, i < j → L i j = 0)
    (h1 : ∀ i, L i i = 1) (hU : ∀ i j, j < i → U i j = 0) :
    (L * U).det = ∏ i, U i i := by
  rw [det_mul]
  rw [det_L L hL h1]
  rw [one_mul]
  exact det_of_isUpperTriangular (fun i j hij => hU i j hij)

end Κεφ3.OrizousaLU
