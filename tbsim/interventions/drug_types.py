"""TB drug type definitions and parameters."""

__all__ = ['drug_params']

# Drug parameters keyed by drug type name.
# Each entry maps to a dict of clinical parameters.
drug_params = dict(
    dots                = dict(cure_prob=0.85, inactivation_rate=0.10, resistance_rate=0.02, relapse_rate=0.05, mortality_rate=0.80, duration=180, adherence_rate=0.85, cost_per_course=100),
    dots_improved       = dict(cure_prob=0.90, inactivation_rate=0.08, resistance_rate=0.015, relapse_rate=0.03, mortality_rate=0.85, duration=180, adherence_rate=0.90, cost_per_course=150),
    empiric_treatment   = dict(cure_prob=0.70, inactivation_rate=0.15, resistance_rate=0.05, relapse_rate=0.10, mortality_rate=0.60, duration=90,  adherence_rate=0.75, cost_per_course=80),
    first_line_combo    = dict(cure_prob=0.95, inactivation_rate=0.05, resistance_rate=0.01, relapse_rate=0.02, mortality_rate=0.90, duration=120, adherence_rate=0.88, cost_per_course=200),
    second_line_combo   = dict(cure_prob=0.75, inactivation_rate=0.12, resistance_rate=0.03, relapse_rate=0.08, mortality_rate=0.70, duration=240, adherence_rate=0.80, cost_per_course=500),
    third_line_combo    = dict(cure_prob=0.60, inactivation_rate=0.20, resistance_rate=0.08, relapse_rate=0.15, mortality_rate=0.50, duration=360, adherence_rate=0.70, cost_per_course=1000),
    latent_treatment    = dict(cure_prob=0.90, inactivation_rate=0.02, resistance_rate=0.005, relapse_rate=0.01, mortality_rate=0.95, duration=90,  adherence_rate=0.85, cost_per_course=50),
)
