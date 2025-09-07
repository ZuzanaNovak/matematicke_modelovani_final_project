def Z1_glucose_input(t: float, BW_kg: float) -> float:
    """IVGTT: 0.33 g/kg over 3 min → mg/min."""
    total_mg = 0.33 * BW_kg * 1000.0
    return total_mg/3.0 if (0.0 <= t < 3.0) else 0.0

def Z2_insulin_input(t: float) -> float:
    """No exogenous insulin in IVGTT."""
    return 0.0
