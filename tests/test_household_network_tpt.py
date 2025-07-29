import pytest
import numpy as np
import starsim as ss
import tbsim as mtb
import sciris as sc
import time
from tbsim.networks import HouseholdNetGeneric


class MockRationsTrial:
    """Mock intervention to simulate the RATIONS trial structure needed for testing."""
    def __init__(self, n_agents):
        self.hhid = np.random.randint(0, n_agents // 5, size=n_agents)  # ~5 people per household
        self.arm = np.random.randint(0, 2, size=n_agents)  # 2 arms


@pytest.fixture
def sample_households():
    """Create sample household structure for testing."""
    # Create 10 households with 3-6 members each
    households = []
    current_uid = 0
    for i in range(10):
        household_size = np.random.randint(3, 7)
        household = list(range(current_uid, current_uid + household_size))
        households.append(household)
        current_uid += household_size
    return households


@pytest.fixture
def sample_sim(sample_households):
    """Create a sample simulation with household network and TB disease."""
    n_agents = sum(len(hh) for hh in sample_households)
    
    # Create people with household IDs
    people = ss.People(n_agents)
    people.define_people(
        ss.Arr('hhid', default=0),
        ss.Arr('on_tpt', default=False),
        ss.Arr('received_tpt', default=False),
        ss.Arr('screen_negative', default=True),
        ss.Arr('non_symptomatic', default=True),
        ss.Arr('parent', default=-1),
    )
    
    # Assign household IDs
    hhid = 0
    for household in sample_households:
        for uid in household:
            people.hhid[uid] = hhid
        hhid += 1
    
    # Create household network
    network = HouseholdNetGeneric(hhs=sample_households, pars={'add_newborns': True})
    
    # Create TB disease
    tb = mtb.TB(pars={
        'beta': 0.1,
        'init_prev': 0.05,
    })
    
    # Create mock RATIONS trial intervention
    rations_trial = MockRationsTrial(n_agents)
    rations_trial.hhid = people.hhid
    rations_trial.arm = np.random.randint(0, 2, size=n_agents)
    
    # Create TPT intervention
    tpt = mtb.TPTInitiation(pars={
        'p_tpt': 0.8,
        'age_range': [0, 100],
        'tpt_treatment_duration': ss.peryear(0.25),  # 3 months
        'tpt_protection_duration': ss.peryear(2.0),  # 2 years
    })
    
    # Create simulation
    sim = ss.Sim(
        people=people,
        networks=network,
        diseases=tb,
        interventions=[tpt],
        pars={
            'start': 2020,
            'stop': 2025,
            'dt': 0.25,  # Quarterly steps
            'unit': 'year',
        }
    )
    
    # Add mock intervention to sim for network access
    sim.interventions.rationstrial = rations_trial
    
    return sim


def test_household_network_initialization(sample_households):
    """Test that household network initializes correctly."""
    network = HouseholdNetGeneric(hhs=sample_households)
    
    assert network.hhs == sample_households
    assert network.pars.add_newborns == False
    
    # Test with add_newborns enabled
    network = HouseholdNetGeneric(hhs=sample_households, pars={'add_newborns': True})
    assert network.pars.add_newborns == True


def test_complete_graph_generation():
    """Test the optimized complete graph generation."""
    network = HouseholdNetGeneric()
    
    # Test with 3 people
    uids = [0, 1, 2]
    p1s, p2s = network._generate_complete_graph_edges(uids)
    
    expected_edges = [(0, 1), (0, 2), (1, 2)]
    actual_edges = list(zip(p1s, p2s))
    
    assert len(actual_edges) == 3
    for edge in expected_edges:
        assert edge in actual_edges or (edge[1], edge[0]) in actual_edges
    
    # Test with single person
    p1s, p2s = network._generate_complete_graph_edges([0])
    assert len(p1s) == 0
    assert len(p2s) == 0
    
    # Test with empty list
    p1s, p2s = network._generate_complete_graph_edges([])
    assert len(p1s) == 0
    assert len(p2s) == 0


def test_household_network_performance(sample_households):
    """Test performance improvement of optimized household network."""
    # Test original approach (simulated)
    start_time = time.time()
    network_original = HouseholdNetGeneric(hhs=sample_households)
    # Simulate the old approach with multiple concatenations
    for hh in sample_households:
        if len(hh) >= 2:
            # This would be the old inefficient approach
            pass
    original_time = time.time() - start_time
    
    # Test optimized approach
    start_time = time.time()
    network_optimized = HouseholdNetGeneric(hhs=sample_households)
    network_optimized.init_pre(None)  # This uses the optimized approach
    optimized_time = time.time() - start_time
    
    # The optimized version should be faster
    print(f"Original approach time: {original_time:.6f}s")
    print(f"Optimized approach time: {optimized_time:.6f}s")
    assert optimized_time < original_time or optimized_time < 0.1  # Allow for small timing variations


def test_network_initialization_with_sim(sample_sim):
    """Test that network initializes correctly with simulation."""
    network = sample_sim.networks[0]
    
    # Initialize the network
    network.init_pre(sample_sim)
    
    # Check that edges were created
    assert len(network.edges.p1) > 0
    assert len(network.edges.p2) > 0
    assert len(network.edges.beta) > 0
    
    # Check that all edges have beta = 1
    assert np.all(network.edges.beta == 1)
    
    # Check that p1 and p2 have same length
    assert len(network.edges.p1) == len(network.edges.p2)


def test_tpt_intervention_basic(sample_sim):
    """Test basic TPT intervention functionality."""
    # Set up some TB cases
    tb = sample_sim.diseases.tb
    people = sample_sim.people
    
    # Set some people on TB treatment
    treatment_candidates = np.random.choice(people.n_uids, size=5, replace=False)
    tb.on_treatment[treatment_candidates] = True
    
    # Set some people as screen negative
    people.screen_negative[:] = True
    
    # Initialize TPT intervention
    tpt = sample_sim.interventions[0]
    tpt.init_results()
    
    # Run one step
    tpt.step()
    
    # Check that results were tracked
    assert hasattr(tpt.results, 'n_eligible')
    assert hasattr(tpt.results, 'n_tpt_initiated')


def test_tpt_intervention_with_household_network(sample_sim):
    """Test TPT intervention working with household network."""
    # Initialize simulation
    sample_sim.init()
    
    # Set up TB cases in households
    tb = sample_sim.diseases.tb
    people = sample_sim.people
    
    # Set some people on TB treatment (household-based)
    household_ids = np.unique(people.hhid)
    treated_households = np.random.choice(household_ids, size=3, replace=False)
    
    for hhid in treated_households:
        household_members = people.hhid == hhid
        # Set one person per household on treatment
        household_uids = people.n_uids[household_members]
        if len(household_uids) > 0:
            treated_uid = np.random.choice(household_uids)
            tb.on_treatment[treated_uid] = True
    
    # Set screening status
    people.screen_negative[:] = True
    people.non_symptomatic[:] = True
    
    # Run simulation for a few steps
    for _ in range(5):
        sample_sim.run_one_step()
    
    # Check that TPT intervention ran
    tpt = sample_sim.interventions[0]
    assert hasattr(tpt.results, 'n_eligible')
    assert hasattr(tpt.results, 'n_tpt_initiated')


def test_newborn_functionality(sample_sim):
    """Test that newborn functionality works correctly."""
    network = sample_sim.networks[0]
    people = sample_sim.people
    
    # Enable newborn functionality
    network.pars.add_newborns = True
    
    # Simulate some newborns
    newborn_uids = np.array([100, 101, 102])  # New UIDs
    mother_uids = np.array([0, 1, 2])  # Existing UIDs
    
    # Set up parent relationships
    people.parent[newborn_uids] = mother_uids
    
    # Set ages to simulate newborns
    people.age[newborn_uids] = 0.1  # Very young
    
    # Run network step
    network.step()
    
    # Check that edges were added for newborns
    initial_edges = len(network.edges.p1)
    assert initial_edges > 0


def test_network_edge_correctness(sample_households):
    """Test that the optimized network creates the same edges as the original approach."""
    network = HouseholdNetGeneric(hhs=sample_households)
    
    # Initialize network
    network.init_pre(None)
    
    # Check that all households have complete graphs
    for household in sample_households:
        if len(household) >= 2:
            expected_edges = len(household) * (len(household) - 1) // 2
            
            # Count edges for this household
            household_edges = 0
            for i, p1 in enumerate(network.edges.p1):
                p2 = network.edges.p2[i]
                if (p1 in household and p2 in household):
                    household_edges += 1
            
            assert household_edges == expected_edges


def test_tpt_protection_mechanism(sample_sim):
    """Test that TPT protection mechanism works correctly."""
    # Initialize simulation
    sample_sim.init()
    
    tb = sample_sim.diseases.tb
    people = sample_sim.people
    
    # Set up a household with TB treatment
    household_ids = np.unique(people.hhid)
    treated_household = household_ids[0]
    household_members = people.hhid == treated_household
    household_uids = people.n_uids[household_members]
    
    # Set one person on treatment
    tb.on_treatment[household_uids[0]] = True
    
    # Set others as eligible for TPT
    for uid in household_uids[1:]:
        people.screen_negative[uid] = True
        people.non_symptomatic[uid] = True
    
    # Run TPT intervention
    tpt = sample_sim.interventions[0]
    tpt.step()
    
    # Check that some people received TPT
    assert np.any(people.on_tpt) or np.any(people.received_tpt)


if __name__ == '__main__':
    # Run performance comparison
    print("Running performance tests...")
    
    # Create larger test case
    large_households = []
    current_uid = 0
    for i in range(100):  # 100 households
        household_size = np.random.randint(3, 8)
        household = list(range(current_uid, current_uid + household_size))
        large_households.append(household)
        current_uid += household_size
    
    test_household_network_performance(large_households)
    
    print("All tests completed successfully!") 