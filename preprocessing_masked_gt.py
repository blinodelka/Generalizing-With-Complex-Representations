#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: marinadubova
"""

import pandas as pd
import numpy as np
import scipy.stats as ss


# class for unpickling    
class multivariate_gaussian:

    def __init__(self, n_dims, max_loc, wishart_scale=5):

        self.n_dims = n_dims
        self.loc = np.random.uniform(max_loc, size=n_dims)

        # Change wishart_scale (default = 5) to a bigger number to get less elongated distributions
        self.scale = ss.wishart.rvs(scale=np.eye(n_dims) * wishart_scale, df=self.n_dims + 2, size=1)

    def sample(self):

        return ss.multivariate_normal.rvs(mean=self.loc, cov=self.scale)

    def sample_conditioned(self, fixed_dims, values, return_full=False):
        assert len(fixed_dims) < self.n_dims, "Too many fixed dimensions!"

        dims_set = set(fixed_dims)
        free_dims = [i for i in range(self.n_dims) if i not in dims_set]


        sigma_22 = self.scale[fixed_dims, :][:, fixed_dims]
        sigma_11 = self.scale[free_dims, :][:, free_dims]
        sigma_12 = self.scale[free_dims, :][:, fixed_dims]

        mu_1 = self.loc[free_dims]
        mu_2 = self.loc[fixed_dims]

        tmp = sigma_12 @ np.linalg.inv(sigma_22)
        sigma_bar = sigma_11 - tmp @ sigma_12.T
        mubar = mu_1 + tmp @ (values - mu_2)

        sample = ss.multivariate_normal.rvs(mean=mubar, cov=sigma_bar)

        if return_full:
            res = np.zeros(self.n_dims)
            res[free_dims] = sample
            res[fixed_dims] = values
            return res

        else:
            return sample

    def marginal_pdf(self, dims, values):
        sigma_new = self.scale[dims, :][:, dims]
        mu_new = self.loc[dims]

        return ss.multivariate_normal.pdf(values, mean=mu_new, cov=sigma_new)


class clustered_multivariate_gaussian:

    def __init__(self, n_dims, max_loc, num_clusters, wishart_scale=5):

        self.n_dims = n_dims
        self.max_loc = max_loc
        self.num_clusters = num_clusters

        self.cluster_priors = np.full(num_clusters, 1 / num_clusters)
        self.clusters = [multivariate_gaussian(n_dims, max_loc, wishart_scale=wishart_scale) for _ in range(num_clusters)]

    def sample(self, cluster_probs=None, cond_dims=None, cond_vals=None, return_full=False):

        cluster_ind = np.random.choice(np.arange(self.num_clusters),
                                       p=self.cluster_priors if cluster_probs is None else cluster_probs)
        if cond_dims is not None:
            return self.clusters[cluster_ind].sample_conditioned(cond_dims, cond_vals, return_full)
        else:
            return self.clusters[cluster_ind].sample()

    def sample_conditioned(self, fixed_dims, values, return_full=False):

        posteriors = self.cluster_priors * np.array([c.marginal_pdf(fixed_dims, values) for c in self.clusters])
        posteriors = posteriors / np.sum(posteriors)

        return self.sample(cluster_probs=posteriors, cond_dims=fixed_dims,
                                                     cond_vals=values,
                                                     return_full=return_full)




def convert_dimlists(dimsvals, return_full=True):
    assert isinstance(dimsvals, list), "Dims and values are not a list!"

    return {"cond_dims": None, "cond_vals": None} if not dimsvals else {"cond_dims": np.array(dimsvals[0]),
                                                                        "cond_vals": np.array(dimsvals[1]),
                                                                        "return_full": return_full}
    
# use the following line to specify preprocessing batch number        
#sim_n = int(sys.argv[1])

# here: specifying the batch number manually
sim_n=0

dataset = []
for i in range(sim_n*20,sim_n*20+20):
    # preprocessing of data for each simulation in a batch
    try:
        # CHANGE TO THE PATH WITH YOUR SIMULATION RESULTS 
        data = pd.read_pickle(r'SIM_RESULTS_PATH/{}.pkl'.format(i))
    except:
        pass
        
    else:
        observation = []
        
        # test performance at step 0; here min and average are the same because simulations involved only one agent
        observation.append(np.mean(data[1][0]))
        observation.append(np.min(data[1][0]))         
        # test and train performances at all other timesteps (again, min and average are the same) 
        for k in range(len(data[0])):
    	    observation.append(np.mean(data[0][k]))
    	    observation.append(np.mean(data[1][k+1]))
    	    observation.append(np.min(data[0][k]))
    	    observation.append(np.min(data[1][k+1]))
        # adding parameters of the simulation
        observation.append(data[5]["n_dimensions"])
        observation.append(data[5]["n_clusters"])
        observation.append(data[5]["experimentation_strategy"])
        observation.append(data[5]["explanation_capacity"])
        observation.append(data[5]['measurement_capacity'])
        observation.append(data[5]['n_scientists'])
        observation.append(data[5]['collective_strategy'])
	    			
        observation.append(np.nan)
        observation.append(np.nan)
		
        # empty variables here because simulation involved only one agent
        mean_dist, mean_dist_over_time, ind_dist_trend = np.nan, np.nan, np.nan
        ba_dist, ba_dist_over_time, ba_dist_trend = np.nan, np.nan, np.nan
        med_pdf, med_pdf_over_time, pdf_trend = np.nan, np.nan, np.nan
        pred_disagreement = np.nan
        observation.append(mean_dist)
        observation.append(mean_dist_over_time)
        observation.append(ind_dist_trend)
        observation.append(ba_dist)
        observation.append(ba_dist_over_time)
        observation.append(ba_dist_trend)
        observation.append(med_pdf)
        observation.append(med_pdf_over_time)
        observation.append(pdf_trend)
        observation.append(pred_disagreement)
        
        #adding this simulation's data to the batch's dataset
        dataset.append(observation)   

		
# specifying column names for the data
column_names = []
column_names.append("average_global_performance_0")
column_names.append("best_global_score_0")

for l in range(len(data[0])):
    column_names.append("average_local_performance_{}".format((l+1)*(300//len(data[0]))))
    column_names.append("average_global_performance_{}".format((l+1)*(300//len(data[0]))))
    column_names.append("best_local_score_{}".format((l+1)*(300//len(data[0]))))
    column_names.append("best_global_score_{}".format((l+1)*(300//len(data[0]))))

column_names.extend(['n_dimensions', 'n_clusters', 'experimentation_strategy',
                                             'explanation_capacity', 'measurement_capacity', 'n_agents', 'collective_strategy',
                                             'agent_explanation_divergence', 
                                             'pretraining_steps',
                                             'ind_mean_distance_of_samples', 'ind_mean_distance_of_samples_over_time', 'ind_mean_distance_of_samples_trend',
                                             'ba_mean_distance_of_samples', 'ba_mean_distance_of_samples_over_time', 'ba_mean_distance_of_samples_trend',
                                             'median_pdf_of_samples', 'median_pdf_of_samples_over_time', 'median_pdf_of_samples_trend',
                                             'disagreement_in_theoretical_predictions'])

data_sims = pd.DataFrame(dataset, columns = column_names)

# CHANGE TO YOUR OUTPUT DATA PATH 
data_sims.to_csv("OUTPUT_DATA_PATH/sim_45_{}.csv".format(sim_n))
