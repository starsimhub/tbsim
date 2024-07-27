import nutrition.bmistatus as BmiStatus

# Representing the table as a dictionary where keys are tuples of the initial and final status
bmi_transition_table = {
    (BmiStatus.SEVERE_THINNESS, BmiStatus.SEVERE_THINNESS): 130,
    (BmiStatus.SEVERE_THINNESS, BmiStatus.MODERATE_THINNESS): 66,
    (BmiStatus.SEVERE_THINNESS, BmiStatus.MILD_THINNESS): 14,
    (BmiStatus.SEVERE_THINNESS, BmiStatus.NORMAL_WEIGHT): 1,
    (BmiStatus.SEVERE_THINNESS, BmiStatus.ABOVE_NORMAL_WEIGHT): 0,
    (BmiStatus.MODERATE_THINNESS, BmiStatus.SEVERE_THINNESS): 14,
    (BmiStatus.MODERATE_THINNESS, BmiStatus.MODERATE_THINNESS): 91,
    (BmiStatus.MODERATE_THINNESS, BmiStatus.MILD_THINNESS): 93,
    (BmiStatus.MODERATE_THINNESS, BmiStatus.NORMAL_WEIGHT): 8,
    (BmiStatus.MODERATE_THINNESS, BmiStatus.ABOVE_NORMAL_WEIGHT): 0,
    (BmiStatus.MILD_THINNESS, BmiStatus.SEVERE_THINNESS): 4,
    (BmiStatus.MILD_THINNESS, BmiStatus.MODERATE_THINNESS): 23,
    (BmiStatus.MILD_THINNESS, BmiStatus.MILD_THINNESS): 280,
    (BmiStatus.MILD_THINNESS, BmiStatus.NORMAL_WEIGHT): 138,
    (BmiStatus.MILD_THINNESS, BmiStatus.ABOVE_NORMAL_WEIGHT): 0,
    (BmiStatus.NORMAL_WEIGHT, BmiStatus.SEVERE_THINNESS): 0,
    (BmiStatus.NORMAL_WEIGHT, BmiStatus.MODERATE_THINNESS): 0,
    (BmiStatus.NORMAL_WEIGHT, BmiStatus.MILD_THINNESS): 34,
    (BmiStatus.NORMAL_WEIGHT, BmiStatus.NORMAL_WEIGHT): 1611,
    (BmiStatus.NORMAL_WEIGHT, BmiStatus.ABOVE_NORMAL_WEIGHT): 74,
    (BmiStatus.ABOVE_NORMAL_WEIGHT, BmiStatus.SEVERE_THINNESS): 0,
    (BmiStatus.ABOVE_NORMAL_WEIGHT, BmiStatus.MODERATE_THINNESS): 0,
    (BmiStatus.ABOVE_NORMAL_WEIGHT, BmiStatus.MILD_THINNESS): 0,
    (BmiStatus.ABOVE_NORMAL_WEIGHT, BmiStatus.NORMAL_WEIGHT): 6,
    (BmiStatus.ABOVE_NORMAL_WEIGHT, BmiStatus.ABOVE_NORMAL_WEIGHT): 237,
}

# Print the table
for (initial_status, final_status), value in bmi_transition_table.items():
    print(f"From {initial_status.name} to {final_status.name}: {value}")