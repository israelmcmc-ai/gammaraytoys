import numpy as np

def poisson_binned_log_likelihood(data, expectation):

    # Same as this 
    #log_like = np.sum(np.log(np.power(expectation, data) * np.exp(-expectation) / factorial(data)))
    # The factorial is taken out since it's a constant and only likelihood *differences* matter
    
    log_like = np.nansum(data*np.log(expectation) - expectation)
    
    return log_like

def unbinned_log_likelihood(expectation_density, total_expectation):

    """
    Extended unbinned Poisson log-likelihood.

    expectation_density = expectation density -- expected events per unit of
        measured phase space -- evaluated at each observed data point
    total_expectation = total expected number of events, i.e. the integral
        of the expectation density over the full measured phase space
    """

    log_like = -total_expectation + np.sum(np.log(expectation_density))

    return log_like

    
