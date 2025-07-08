import starsim as ss 
import tbsim as mtb 
import numpy as np 
import pandas as pd
import os

def make_tb(pars = None):
    """
    Set up the TB simulation with default parameters.
    """
    # Define simulation parameters
    if pars is None: 
        pars = dict(
            beta = ss.rate_prob(0.1),
            init_prev = ss.bernoulli(p=0.25),
            unit = 'day'
        )
    return mtb.TB(pars=pars)


def make_pop(pars = None, n_agents=500):
    """
    Set up the population with default parameters.
    """
    # Define population parameters
    if pars is None: 
        pars = dict(
            n_agents=n_agents,  # Number of agents in the population
            age_date=load_age_data('default'),  # Load age data from a CSV file or use default data
        )
        
    pop = ss.People(
        n_agents = n_agents,
        age_data = load_age_data,
    )
    return pop

def make_net(pars = None):
    """
    Set up the network with default parameters.
    """
    # Define network parameters
    if pars is None: 
        pars = dict(
            n_contacts=ss.poisson(lam=5),  # Number of contacts per agent
            dur=0,  # Duration of contacts
            # Add any other default parameters here
        )
    return ss.RandomNet(pars=pars)


def make_births(pars = None):
    """
    Set up the births demographic with default parameters.
    """
    # Define births parameters
    if pars is None: 
        pars = dict(
            birth_rate=15,  # Birth rate
            # Add any other default parameters here
        )
    return ss.Births(pars=pars)


def make_deaths(pars = None):
    """
    Set up the deaths demographic with default parameters.
    """
    # Define deaths parameters
    if pars is None: 
        pars = dict(
            death_rate=15,  # Death rate
            # Add any other default parameters here
        )
    return ss.Deaths(pars=pars)

def make_intervention(pars = None):
    """
    Set up the intervention with default parameters.
    """
    # Define intervention parameters
    if pars is None: 
        pars = dict(    )
    return ss.Intervention(pars=pars)  # Placeholder for the actual intervention class

def make_hiv(pars = None):
    """
    Set up the HIV intervention with default parameters.
    """
    # Define HIV parameters
    if pars is None: 
        pars = dict(
            # Add any default parameters here
        )
    return ss.HIV(pars=pars)  # Placeholder for the actual HIV class

def make_cnn(pars = None):
    """
    Set up the CNN intervention with default parameters.
    """
    # Define CNN parameters
    if pars is None: 
        return dict() #TODO: This may need to be updated with more specific cases of CNN, for instance, TB - Malnutrition, TB - HIV, etc.
    return ss.CNN(pars=pars)  # Placeholder for the actual CNN class

def load_age_data(source='default', file_path=''):
    """
    Load population data from a CSV file or use default data.
    """
    if source == 'default':
        # Default population data
        # Gathered from WPP, https://population.un.org/wpp/Download/Standard/MostUsed/
        age_data = pd.DataFrame({ 
            'age': np.arange(0, 101, 5),
            'value': [5791, 4446, 3130, 2361, 2279, 2375, 2032, 1896, 1635, 1547, 1309, 1234, 927, 693, 460, 258, 116, 36, 5, 1, 0]  # 1960
        })
    elif source == 'json':
        if not file_path:
            raise ValueError("file_path must be provided when source is 'json'.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        data = pd.read_json(file_path)
        age_data = pd.DataFrame(data)
    else:
        raise ValueError("Invalid source. Use 'default' or 'json'.")
    return age_data

