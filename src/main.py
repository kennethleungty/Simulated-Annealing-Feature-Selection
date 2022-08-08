import pandas as pd
import numpy as np
import random
from .utils import train_model

# Load data
PATH = './data/processed/'
X_train = pd.read_csv(PATH + 'X_train.csv')
y_train = pd.read_csv(PATH + 'y_train.csv')

def simulated_annealing(X_train,
                        y_train,
                        maxiters=50,
                        alpha=0.85,
                        beta=1,
                        T_0=0.95,
                        update_iters=1,
                        temp_reduction='geometric'):
    """
    Function to perform feature selection using simulated annealing
    Inputs:
    X_train - Predictor features
    y_train - Train labels
    maxiters - Maximum number of iterations
    alpha - factor to reduce temperature
    beta - constant in probability estimate 
    T_0 - Initial temperature
    update_iters - Number of iterations required to update temperature

    Output:
    1) Dataframe of the parameters explored and corresponding model performance
    2) Best metric score (i.e. AUC score for this case)
    3) List of subset features that correspond to the best metric
    """
    columns = ['Iteration', 'Feature Count', 'Feature Set', 
               'Metric', 'Best Metric', 'Acceptance Probability', 
               'Random Number', 'Outcome']
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_subset = None
    hash_values = set()
    T = T_0

    # Get ascending range indices of all columns
    full_set = set(np.arange(len(X_train.columns)))

    # Generate initial random subset based on ~50% of columns
    curr_subset = set(random.sample(full_set, round(0.5 * len(full_set))))

    # Get baseline metric score (i.e. AUC) of initial random subset
    X_curr = X_train.iloc[:, list(curr_subset)]
    prev_metric = train_model(X_curr, y_train)
    best_metric = prev_metric

    for i in range(maxiters):
        # Termination conditions
        if T < 0.01:
            print(f'Temperature {T} below threshold. Termination condition met')
            break
        
        print(f'Starting Iteration {i+1}')
        # Decide what type of pertubation to make
        if len(curr_subset) == len(full_set): 
            move = 'Remove'
        elif len(curr_subset) == 2: # Not to go below 2 features
            move = random.choice(['Add', 'Replace'])
        else:
            move = random.choice(['Add', 'Replace', 'Remove'])

        # Execute pertubation (i.e. alter current subset to get new subset)
        while True:
            # Get columns not yet used in current subset
            pending_cols = full_set.difference(curr_subset) 
            new_subset = curr_subset.copy()   

            if move == 'Add':        
                new_subset.add(random.choice(list(pending_cols)))
            elif move == 'Replace': 
                new_subset.remove(random.choice(list(curr_subset)))
                new_subset.add(random.choice(list(pending_cols)))
            else:
                new_subset.remove(random.choice(list(curr_subset)))
                
            if new_subset in hash_values:
                'Subset already visited'
            else:
                hash_values.add(frozenset(new_subset)) # !!!! Review whether include here or after modeling !!!
                break

        # Filter dataframe to current subset
        X_new = X_train.iloc[:, list(new_subset)]

        # Get metric of new subset
        metric = train_model(X_new, y_train)

        if metric > prev_metric:
            print('Local improvement in metric from {:8.4f} to {:8.4f} '
                  .format(prev_metric, metric) + ' - parameters accepted')
            outcome = 'Improved'
            accept_prob, rnd = '-', '-'
            prev_metric = metric
            curr_subset = new_subset.copy()

            # Keep track of overall best metric so far
            if metric > best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} '
                      .format(best_metric, metric) + ' - best parameters updated')
                best_metric = metric
                best_subset = new_subset.copy()
                
        else:
            rnd = np.random.uniform()
            diff = metric - prev_metric
            accept_prob = np.exp(beta * diff / T)

            if rnd < accept_prob:
                print('New subset has worse performance but still accept. Metric change' +
                      ': {:8.4f} acceptance probability: {:6.4f} random number: {:6.4f}'
                      .format(diff, accept_prob, rnd))
                outcome = 'Accept'
                prev_metric = metric
                curr_subset = new_subset.copy()
            else:
                print('New subset has worse performance, therefore reject. Metric change' +
                      ': {:8.4f} acceptance probability: {:6.4f} random number: {:6.4f}'
                      .format(diff, accept_prob, rnd))
                outcome = 'Reject'

        # Update results dataframe
        results.loc[i, 'Iteration'] = i+1
        results.loc[i, 'Feature Count'] = len(curr_subset)
        results.loc[i, 'Feature Set'] = list(curr_subset)
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric
        results.loc[i, 'Acceptance Probability'] = accept_prob
        results.loc[i, 'Random Number'] = rnd
        results.loc[i, 'Outcome'] = outcome

        # Temperature cooling schedule
        if i % update_iters == 0:
            if temp_reduction == 'geometric':
                T = alpha * T
            elif temp_reduction == 'linear':
                T -= alpha
            elif temp_reduction == 'slow decrease':
                b = 5 # Arbitrary constant
                T = T / (1 + b * T)
            else:
                raise Exception("Temperature reduction strategy not recognized")

    # Drop NaN rows in results
    results = results.dropna(axis=0, how='all')

    # Convert column indices of best subset to original names
    best_subset_cols = [list(X_train.columns)[i] for i in list(best_subset)]

    return results, best_metric, best_subset_cols

# Adapted from: https://github.com/santhoshhari/simulated_annealing/blob/master/simulated_annealing.py
