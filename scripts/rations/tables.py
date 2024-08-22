from tbsim.nutritionenums import eBmiStatus

# Representing the table as a dictionary where keys are tuples of the initial and final status
bmi_transition_table = {
    (eBmiStatus.SEVERE_THINNESS, eBmiStatus.SEVERE_THINNESS): 130,
    (eBmiStatus.SEVERE_THINNESS, eBmiStatus.MODERATE_THINNESS): 66,
    (eBmiStatus.SEVERE_THINNESS, eBmiStatus.MILD_THINNESS): 14,
    (eBmiStatus.SEVERE_THINNESS, eBmiStatus.NORMAL_WEIGHT): 1,
    (eBmiStatus.SEVERE_THINNESS, eBmiStatus.OVERWEIGHT): 0,
    (eBmiStatus.MODERATE_THINNESS, eBmiStatus.SEVERE_THINNESS): 14,
    (eBmiStatus.MODERATE_THINNESS, eBmiStatus.MODERATE_THINNESS): 91,
    (eBmiStatus.MODERATE_THINNESS, eBmiStatus.MILD_THINNESS): 93,
    (eBmiStatus.MODERATE_THINNESS, eBmiStatus.NORMAL_WEIGHT): 8,
    (eBmiStatus.MODERATE_THINNESS, eBmiStatus.OVERWEIGHT): 0,
    (eBmiStatus.MILD_THINNESS, eBmiStatus.SEVERE_THINNESS): 4,
    (eBmiStatus.MILD_THINNESS, eBmiStatus.MODERATE_THINNESS): 23,
    (eBmiStatus.MILD_THINNESS, eBmiStatus.MILD_THINNESS): 280,
    (eBmiStatus.MILD_THINNESS, eBmiStatus.NORMAL_WEIGHT): 138,
    (eBmiStatus.MILD_THINNESS, eBmiStatus.OVERWEIGHT): 0,
    (eBmiStatus.NORMAL_WEIGHT, eBmiStatus.SEVERE_THINNESS): 0,
    (eBmiStatus.NORMAL_WEIGHT, eBmiStatus.MODERATE_THINNESS): 0,
    (eBmiStatus.NORMAL_WEIGHT, eBmiStatus.MILD_THINNESS): 34,
    (eBmiStatus.NORMAL_WEIGHT, eBmiStatus.NORMAL_WEIGHT): 1611,
    (eBmiStatus.NORMAL_WEIGHT, eBmiStatus.OVERWEIGHT): 74,
    (eBmiStatus.OVERWEIGHT, eBmiStatus.SEVERE_THINNESS): 0,
    (eBmiStatus.OVERWEIGHT, eBmiStatus.MODERATE_THINNESS): 0,
    (eBmiStatus.OVERWEIGHT, eBmiStatus.MILD_THINNESS): 0,
    (eBmiStatus.OVERWEIGHT, eBmiStatus.NORMAL_WEIGHT): 6,
    (eBmiStatus.OVERWEIGHT, eBmiStatus.OVERWEIGHT): 237,
}

# Print the table
for (initial_status, final_status), value in bmi_transition_table.items():
    print(f"From {initial_status.name} to {final_status.name}: {value}")