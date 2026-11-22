-- Συνοδευτικό ψηφιακό υλικό του βιβλίου: Ματζάκος, Ν. (2026).
-- Στοιχεία Γραμμικής Άλγεβρας, Διαφορικού & Ολοκληρωτικού Λογισμού. NewTech Publications.
-- Κεφάλαιο 0 · Νόμος Ημιτόνων και Συνημιτόνων · ελέγχθηκε με Lean v4.37.0-rc1 και Mathlib (22/11/2026).

import Mathlib

namespace Κεφ0.NomoiTrig
open EuclideanGeometry

theorem cosines (A B C : EuclideanSpace ℝ (Fin 2)) :
    dist A B ^ 2 = dist B C ^ 2 + dist C A ^ 2
      - 2 * dist B C * dist C A * Real.cos (∠ A C B) := by
  have h := law_cos A C B
  rw [dist_comm A C] at h
  linear_combination h

theorem sines (A B C : EuclideanSpace ℝ (Fin 2)) :
    dist B C * Real.sin (∠ A B C) = dist C A * Real.sin (∠ B A C) := by
  have h := law_sin A B C
  rw [angle_comm C A B] at h
  linear_combination h

end Κεφ0.NomoiTrig
