import numpy as np
def plot_with_errorband(ax,trial_data,indices = None,label='',stepped=False):
    '''
        Depending on whether we want to plot step function curves or linearly-iterploated curves, 
        set stepped as True or False.
        
        It makes more sense to plot step function curves for expected regret of M's, 
        whereas plotting linearly-interpolated curves is more appropriate for the realized regret.
    '''
    mu = np.mean(trial_data,axis=0)
    std = np.std(trial_data,axis=0)

    if indices is None:
        indices = range(len(mu))
    if stepped:
        mu = np.repeat(mu,2)[1:]
        std = np.repeat(std,2)[1:]
        indices = np.repeat(indices,2)[:-1]
   
    ax.plot(indices,np.array(mu),label=label)
    ax.fill_between(indices,mu+std,mu-std,alpha=0.2)

def plot_with_percentile(ax,trial_data,l_percentile,u_percentile,indices = None,label=''):
    median = np.median(trial_data,axis=0)

    if indices is None:
        indices = np.arange(len(median))
    
    ax.plot(indices,np.array(median),label=label)
    ax.fill_between(indices,np.percentile(trial_data,l_percentile,axis=0),\
                    np.percentile(trial_data,u_percentile,axis=0)\
                    ,alpha=0.2)