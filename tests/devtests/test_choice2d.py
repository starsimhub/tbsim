"""
Test tbsim.choice2d() -- a 2D version of ss.choice() allowing per-agent probabilities.
"""

import numpy as np
import sciris as sc
import tbsim

sc.options(interactive=False)


def test_basic():
    """ Test basic choice2d with per-agent probabilities """
    sc.heading('Testing basic choice2d')

    uids = np.arange(6)
    p = np.array([
        [0.1, 0.5, 0.4],
        [0.2, 0.3, 0.5],
        [0.3, 0.4, 0.3],
        [0.4, 0.2, 0.4],
        [0.5, 0.3, 0.2],
        [0.8, 0.1, 0.1],
    ])

    d = tbsim.choice2d(p=p, strict=False).init(slots=uids)
    draws = d.rvs(uids)

    print(f'choice2d draws: {draws}')
    assert len(draws) == len(uids), f'Expected {len(uids)} draws, got {len(draws)}'
    assert all(d in [0, 1, 2] for d in draws), f'All draws should be in [0, 1, 2], got {draws}'
    return draws


def test_statistical():
    """ Test that choice2d produces statistically correct results """
    sc.heading('Testing choice2d statistical properties')

    n = 100_000
    uids = np.arange(n)

    # All agents have the same probabilities: heavily weighted toward choice 2
    p = np.tile([0.1, 0.2, 0.7], (n, 1))

    d = tbsim.choice2d(p=p, strict=False).init(slots=uids)
    draws = d.rvs(uids)

    # Check empirical frequencies match expected
    for val, expected_freq in [(0, 0.1), (1, 0.2), (2, 0.7)]:
        actual_freq = (draws == val).mean()
        assert np.isclose(actual_freq, expected_freq, atol=0.02), \
            f'Choice {val}: expected freq {expected_freq}, got {actual_freq}'
        print(f'  Choice {val}: expected={expected_freq}, actual={actual_freq:.3f}')

    return draws


def test_custom_values():
    """ Test choice2d with custom choice values via 'a' parameter """
    sc.heading('Testing choice2d with custom values')

    n = 50_000
    uids = np.arange(n)

    a = np.array([10, 20, 30])
    p = np.tile([0.5, 0.3, 0.2], (n, 1))

    d = tbsim.choice2d(a=a, p=p, strict=False).init(slots=uids)
    draws = d.rvs(uids)

    expected_mean = (a * [0.5, 0.3, 0.2]).sum()  # 10*0.5 + 20*0.3 + 30*0.2 = 17
    actual_mean = draws.mean()
    assert np.isclose(actual_mean, expected_mean, rtol=0.05), \
        f'Expected mean ~{expected_mean}, got {actual_mean}'
    print(f'  Expected mean={expected_mean}, actual={actual_mean:.2f}')
    return draws


def test_per_agent_variation():
    """ Test that per-agent probabilities produce different distributions per agent """
    sc.heading('Testing per-agent probability variation')

    uids = np.array([0, 1])

    # Agent 0: always picks choice 0; Agent 1: always picks choice 1
    p = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    d = tbsim.choice2d(p=p, strict=False).init(slots=uids)
    draws = d.rvs(uids)

    print(f'  Deterministic draws: {draws}')
    assert draws[0] == 0, f'Agent 0 should always pick 0, got {draws[0]}'
    assert draws[1] == 1, f'Agent 1 should always pick 1, got {draws[1]}'
    return draws


def test_p_as_positional():
    """ Test that passing a 2D array as the first positional arg is treated as p """
    sc.heading('Testing choice2d(p) shorthand')

    uids = np.arange(4)
    p = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    # Pass 2D array as first positional arg -- should be interpreted as p
    d = tbsim.choice2d(p, strict=False).init(slots=uids)
    draws = d.rvs(uids)

    print(f'  Shorthand draws: {draws}')
    assert draws[0] == 0 and draws[1] == 0
    assert draws[2] == 1 and draws[3] == 1
    return draws


def test_array_params():
    """ Test choice2d with sparse UIDs, analogous to test_array in starsim """
    sc.heading('Testing choice2d with sparse UIDs')

    uids = np.array([1, 3])
    p = np.array([
        [0.9, 0.1],  # Agent 1: strongly prefers choice 0
        [0.1, 0.9],  # Agent 3: strongly prefers choice 1
    ])

    d = tbsim.choice2d(p=p, strict=False).init(slots=np.arange(uids.max() + 1))
    draws = d.rvs(uids)

    print(f'  Array param draws for uids {uids}: {draws}')
    assert len(draws) == len(uids)
    return draws


if __name__ == '__main__':
    sc.options(interactive=True)
    T = sc.timer()

    o1 = test_basic()
    o2 = test_statistical()
    o3 = test_custom_values()
    o4 = test_per_agent_variation()
    o5 = test_p_as_positional()
    o6 = test_array_params()

    T.toc()
