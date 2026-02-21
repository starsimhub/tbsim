import starsim as ss
import numpy as np
from typing import Optional, List, Any, Dict





class TBPeople(ss.People):
    """
    A specialized People class that automatically includes TB-related extra states.
    
    This class inherits from starsim.People and seamlessly adds all TB-related
    states by default, making it easier to create populations for TB simulations
    without manually specifying extra_states.
    
    The class automatically includes 24 TB-related states defined in TBPeople.TB_STATES:
    - Health seeking behavior (sought_care, care_seeking_multiplier)
    - Testing and diagnosis (tested, test_result, diagnosed)
    - Treatment (n_times_treated, treatment_success, treatment_failure)
    - TB symptoms (symptomatic, presymptomatic, non_symptomatic)
    - TB types (eptb, tb_smear)
    - HIV co-infection (hiv_positive)
    - TPT (received_tpt, on_tpt)
    - Household contacts (household_contact, hhid)
    - Vaccination (vaccination_year)
    
    Class Attributes
    ----------------
    TB_STATES : list
        Class-level list of 24 TB-related state objects that are automatically
        included in all TBPeople instances. Can be accessed as TBPeople.TB_STATES.
        
    Parameters
    ----------
    n_agents : int
        Number of agents in the population
    extra_states : list, optional
        Additional extra states to include beyond the default TB states.
        The default TB states will be automatically merged with any provided states.
    age_data : callable or array-like, optional
        Age data for the population. Can be a function that returns age data
        or an array-like object with age information.
    **kwargs
        Additional keyword arguments passed to the parent ss.People class
        
    Instance Attributes
    ------------------
    extra_states : list
        List of all extra states (TB states + any custom states)
    n_agents : int
        Number of agents in the population
        
    Examples
    --------
    Basic usage with default TB states:
    >>> pop = TBPeople(n_agents=1000)
    >>> print(f"Population has {pop.n_agents} agents with {len(pop.extra_states)} states")
    
    With additional custom states:
    >>> custom_states = [ss.FloatArr('SES', default=0.0), ss.BoolState('urban', default=True)]
    >>> pop = TBPeople(n_agents=1000, extra_states=custom_states)
    >>> print(f"Total states: {len(pop.extra_states)}")  # 24 TB + 2 custom = 26
    
    Access TB states directly:
    >>> print(f"TB states available: {len(TBPeople.TB_STATES)}")
    >>> for state in TBPeople.TB_STATES[:3]:
    ...     print(f"  {state.name}")
    
    Use TB states with regular ss.People:
    >>> pop = ss.People(n_agents=100, extra_states=TBPeople.TB_STATES)
    
    Accessing states:
    >>> # Access as attributes
    >>> print(f"Number diagnosed: {pop.diagnosed.sum()}")
    >>> print(f"Number symptomatic: {pop.symptomatic.sum()}")
    
    Creating a simulation:
    >>> import tbsim
    >>> pop = TBPeople(n_agents=1000)
    >>> tb = tbsim.TB(pars={'beta': ss.peryear(0.01), 'init_prev': 0.25})
    >>> sim = ss.Sim(people=pop, diseases=[tb], pars={'dt': ss.days(7)})
    
    Notes
    -----
    - TB states are defined in the class attribute TBPeople.TB_STATES
    - States are accessible both as attributes (pop.sought_care) and via pop.extra_states
    - The class is fully compatible with existing starsim.People functionality
    - Custom states are merged with TB states, not replaced
    - TB_STATES can be accessed without creating an instance
    """
    
    # Class-level TB states definition - encapsulated within the class
    TB_STATES = [
        ss.BoolState('sought_care', default=False),
        ss.FloatArr('care_seeking_multiplier', default=1.0),
        ss.BoolState('multiplier_applied', default=False),
        ss.FloatArr('n_times_tested', default=0.0),
        ss.FloatArr('n_times_treated', default=0.0),
        ss.BoolState('returned_to_community', default=False),
        ss.BoolState('received_tpt', default=False),
        ss.BoolState('tb_treatment_success', default=False),
        ss.BoolState('tested', default=False),
        ss.BoolState('test_result', default=False),
        ss.BoolState('diagnosed', default=False),
        ss.BoolState('on_tpt', default=True),
        ss.BoolState('tb_smear', default=False),
        ss.BoolState('hiv_positive', default=False),
        ss.BoolState('eptb', default=False),
        ss.BoolState('symptomatic', default=False),
        ss.BoolState('presymptomatic', default=False),
        ss.BoolState('non_symptomatic', default=True),
        ss.BoolState('screen_negative', default=True),
        ss.BoolState('household_contact', default=False),
        ss.BoolState('treatment_success', default=False),
        ss.BoolState('treatment_failure', default=False),
        ss.IntArr('hhid', default=-1),
        ss.FloatArr('vaccination_year', default=np.nan),
    ]
    
    def __init__(self, n_agents: int, extra_states: Optional[List[Any]] = None, 
                 age_data: Optional[Any] = None, **kwargs):
        """
        Initialize TBPeople with TB-related states automatically included.
        
        The class automatically includes 24 TB-related states defined in TBPeople.TB_STATES,
        covering health seeking behavior, testing, diagnosis, treatment, symptoms, and more.
        
        Parameters
        ----------
        n_agents : int
            Number of agents in the population
        extra_states : list, optional
            Additional extra states beyond the default TB states.
            These will be merged with TB_STATES, not replaced.
        age_data : callable or array-like, optional
            Age data for the population. Can be a function that returns age data
            or an array-like object with age information.
        **kwargs
            Additional keyword arguments passed to the parent ss.People class
            
        Examples
        --------
        >>> # Basic usage
        >>> pop = TBPeople(n_agents=1000)
        >>> print(f"Population has {pop.n_agents} agents with {len(pop.extra_states)} states")
        
        >>> # With custom states
        >>> custom_states = [ss.FloatArr('SES', default=0.0)]
        >>> pop = TBPeople(n_agents=1000, extra_states=custom_states)
        >>> print(f"Total states: {len(pop.extra_states)}")  # 24 TB + 1 custom = 25
        
        Notes
        -----
        - TB states are defined in the class attribute TBPeople.TB_STATES
        - States are accessible as attributes (pop.diagnosed, pop.symptomatic, etc.)
        - The extra_states parameter merges with TB_STATES, not replaces them
        """
        # Use class-level TB states definition
        tb_states = self.TB_STATES
        
        # Merge with any additional states provided
        if extra_states is not None:
            all_extra_states = tb_states + extra_states
        else:
            all_extra_states = tb_states
        
        # Store the extra states as an instance attribute
        self.extra_states = all_extra_states
        
        # Initialize parent class with merged states
        super().__init__(
            n_agents=n_agents,
            extra_states=all_extra_states,
            age_data=age_data,
            **kwargs
        )
    
    def get_tb_states(self) -> List[Any]:
        """
        Get only the TB-related states (excluding any custom states).
        
        Returns the first 24 states from extra_states, which correspond to the
        TB states defined in TBPeople.TB_STATES.
        
        Returns
        -------
        list
            List of TB-related state objects (same as TBPeople.TB_STATES)
            
        Examples
        --------
        >>> pop = TBPeople(n_agents=100)
        >>> tb_states = pop.get_tb_states()
        >>> print(f"Number of TB states: {len(tb_states)}")
        >>> for state in tb_states:
        ...     print(f"  {state.name}")
        
        >>> # Compare with class attribute
        >>> print(f"Same as TB_STATES: {tb_states is TBPeople.TB_STATES}")
        """
        # Return only the first 24 states (the TB states)
        return self.extra_states[:24]
    
    def get_custom_states(self) -> List[Any]:
        """
        Get only the custom states (excluding TB states).
        
        Returns states beyond the first 24, which correspond to any custom states
        that were added via the extra_states parameter during initialization.
        
        Returns
        -------
        list
            List of custom state objects (empty if no custom states were added)
            
        Examples
        --------
        >>> pop = TBPeople(n_agents=100)
        >>> custom_states = pop.get_custom_states()
        >>> print(f"Number of custom states: {len(custom_states)}")
        
        >>> # With custom states
        >>> custom = [ss.FloatArr('SES', default=0.0)]
        >>> pop = TBPeople(n_agents=100, extra_states=custom)
        >>> print(f"Custom states: {len(pop.get_custom_states())}")  # 1
        """
        # Return states beyond the first 24 (custom states)
        return self.extra_states[24:]
    
    def get_state_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about all states.
        
        Returns a dictionary with state names as keys and metadata as values,
        including type, default value, and whether it's a TB state.
        
        Returns
        -------
        dict
            Dictionary with state names as keys and state info as values.
            Each value contains:
            - 'type': State type (e.g., 'BoolState', 'FloatArr')
            - 'default': Default value for the state
            - 'is_tb_state': Boolean indicating if it's a TB state
            
        Examples
        --------
        >>> pop = TBPeople(n_agents=100)
        >>> info = pop.get_state_info()
        >>> for name, details in info.items():
        ...     print(f"{name}: {details['type']} (default: {details['default']})")
        
        >>> # Check if a state is a TB state
        >>> print(f"sought_care is TB state: {info['sought_care']['is_tb_state']}")
        """
        tb_states = self.get_tb_states()
        state_info = {}
        for i, state in enumerate(self.extra_states):
            state_info[state.name] = {
                'type': type(state).__name__,
                'default': getattr(state, 'default', None),
                'is_tb_state': i < len(tb_states)  # First 24 states are TB states
            }
        return state_info
    
    def list_tb_states(self) -> None:
        """
        Print a formatted list of all TB states with their types and defaults.
        
        Displays a table showing all TB states defined in TBPeople.TB_STATES,
        including their names, types, and default values.
        
        Examples
        --------
        >>> pop = TBPeople(n_agents=100)
        >>> pop.list_tb_states()
        TB States included in TBPeople:
        ==================================================
        sought_care               | BoolState    | default: False
        care_seeking_multiplier   | FloatArr     | default: 1.0
        ...
        """
        print("TB States included in TBPeople:")
        print("=" * 50)
        for state in self.get_tb_states():
            default_val = getattr(state, 'default', 'N/A')
            print(f"{state.name:25} | {type(state).__name__:12} | default: {default_val}")
        print(f"\nTotal TB states: {len(self.get_tb_states())}")
        print(f"Total states (including custom): {len(self.extra_states)}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state of the population.
        
        Returns a dictionary with population statistics including total agents,
        state counts, and counts for key TB states (diagnosed, symptomatic, etc.).
        
        Returns
        -------
        dict
            Dictionary with summary statistics including:
            - 'total_agents': Number of agents in population
            - 'total_states': Total number of states
            - 'tb_states': Number of TB states
            - 'custom_states': Number of custom states
            - 'diagnosed': Count of diagnosed agents
            - 'symptomatic': Count of symptomatic agents
            - 'hiv_positive': Count of HIV positive agents
            - 'tested': Count of tested agents
            - 'sought_care': Count of agents who sought care
            
        Examples
        --------
        >>> pop = TBPeople(n_agents=1000)
        >>> summary = pop.get_state_summary()
        >>> print(f"Diagnosed: {summary['diagnosed']}")
        >>> print(f"Symptomatic: {summary['symptomatic']}")
        >>> print(f"Total agents: {summary['total_agents']}")
        """
        summary = {
            'total_agents': self.n_agents,
            'total_states': len(self.extra_states),
            'tb_states': len(self.get_tb_states()),
            'custom_states': len(self.get_custom_states())
        }
        
        # Add counts for key TB states if they exist
        key_states = ['diagnosed', 'symptomatic', 'hiv_positive', 'tested', 'sought_care']
        for state_name in key_states:
            if hasattr(self, state_name):
                summary[state_name] = getattr(self, state_name).sum()
            else:
                summary[state_name] = 0
                
        return summary
    
    def print_state_summary(self) -> None:
        """
        Print a formatted summary of the population's current state.
        
        Displays a formatted table showing population statistics including
        total agents, state counts, and key TB state counts.
        
        Examples
        --------
        >>> pop = TBPeople(n_agents=1000)
        >>> pop.print_state_summary()
        TBPeople Population Summary:
        ========================================
        Total agents: 1000
        Total states: 24 (24 TB + 0 custom)
        
        Key TB State Counts:
          diagnosed      :    0.0
          symptomatic    :    0.0
          hiv_positive   :    0.0
          tested         :    0.0
          sought_care    :    0.0
        """
        summary = self.get_state_summary()
        
        print("TBPeople Population Summary:")
        print("=" * 40)
        print(f"Total agents: {summary['total_agents']}")
        print(f"Total states: {summary['total_states']} ({summary['tb_states']} TB + {summary['custom_states']} custom)")
        print()
        print("Key TB State Counts:")
        for state_name in ['diagnosed', 'symptomatic', 'hiv_positive', 'tested', 'sought_care']:
            count = summary.get(state_name, 0)
            print(f"  {state_name:15}: {count:6}")
    
    @classmethod
    def get_default_tb_states(cls) -> List[Any]:
        """
        Get the default TB states that would be included in a TBPeople instance.
        
        Returns the class attribute TB_STATES, which contains all 24 TB-related
        state objects that are automatically included in TBPeople instances.
        
        Returns
        -------
        list
            List of default TB state objects (same as TBPeople.TB_STATES)
            
        Examples
        --------
        >>> states = TBPeople.get_default_tb_states()
        >>> print(f"Default TB states: {len(states)}")
        >>> for state in states:
        ...     print(f"  {state.name}")
        
        >>> # Use with regular ss.People
        >>> pop = ss.People(n_agents=100, extra_states=TBPeople.get_default_tb_states())
        
        >>> # Compare with class attribute
        >>> print(f"Same as TB_STATES: {states is TBPeople.TB_STATES}")
        """
        # Return the class-level TB states definition
        return cls.TB_STATES