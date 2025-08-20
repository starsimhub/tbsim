import starsim as ss
import pandas as pd

START = ss.date('1975-01-01')
STOP = ss.date('2030-12-31')

# Simple default parameters
DEFAULT_SPARS = dict(
    dt=ss.days(7),
    start=START,
    stop=STOP,
    rand_seed=123,
    verbose=0,
)

DEFAULT_TBPARS = dict(
    beta=ss.peryear(0.0025),
    init_prev=ss.bernoulli(p=0.25),
    dt=ss.days(7),      
    start=START,
    stop=STOP,
)

# Simple age distribution
AGE_DATA = pd.DataFrame({
    'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]  # Skewed toward younger ages
})
