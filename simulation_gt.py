#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: marinadubova
"""

import scipy.stats as ss
import numpy as np
import keras
from keras import layers
import sys
if sys.version[0] == '3':
    import pickle
else:
    import cPickle as pickle

from keras.callbacks import EarlyStopping
from keras import backend as K

overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = 10)

# classes for the ground truth environments
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
    

# custom loss function for the masked autoencoder -- loss is only calculated for the predictions on the masked dimensions
def my_loss(inputs_w):
    def loss(y_true, y_pred):
        # inputs_w will indicate the masked dimensions; the dimensions with provided dimensions will have 0 weights in the loss function
        return K.abs((y_true-y_pred)*inputs_w)
    return loss


# learning agent 
class scientist:
    def __init__(self, max_dimensions, measurement_capacity, explanation_capacity):
        self.explanation = None # will be replaced with an autoencoder
        self.data_measured = [] # where partial (masked) observations are stored
        self.data_true = [] # where full observations are stored
        self.max_dimensions = max_dimensions # dimensionality of the ground truth
        self.dimension_importance_weights = np.ones((self.max_dimensions)) # all the world's dimensions are equally important
        self.measurement_capacity = measurement_capacity # number of dimensions unmasked for each observation
        self.explanation_capacity = explanation_capacity # width of the autoencoder
        
    def make_observation(self, env):
        
        data_indices = self.pick_dimensions_random("maximum_random") 
        experiment_parameters = []
        raw_observation = np.array(env.sample(**convert_dimlists(experiment_parameters)))
        
        # recording only the dimensions that were measured; unmeasured dimensions get a value of -500
        current_observation = np.zeros((len(raw_observation)))-500 
        current_observation[data_indices] = raw_observation[data_indices]
        
        # saving the masked observation
        self.data_measured.append(current_observation)
        # saving the full observation
        self.data_true.append(raw_observation)

        return current_observation, raw_observation
        
    def pick_dimensions_random(self, strategy): 
        dimensions_to_unmask = []
        dimensions_to_unmask = list(range(self.max_dimensions))
            
        # currently just picking the dimensions randomly at the maximum capacity
        dimensions_to_unmask = np.random.choice(range(self.max_dimensions), 
                                                     size = self.measurement_capacity, 
                                                     p = self.dimension_importance_weights/np.sum(self.dimension_importance_weights), # normalizing here
                                                     replace = False)
            
        return dimensions_to_unmask
    
    def initialize_explanation(self): # initializing autoencoder
        
        encoding_dim = self.explanation_capacity
            
        # input_data -- masked observations
        input_data = keras.Input(shape=(self.max_dimensions,))
        # new vector for input weights for the custom loss function
        inputs_w = keras.Input(shape=(self.max_dimensions,))

        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(encoding_dim, activation='relu')(input_data)
        # "decoded" is the reconstruction of the input
        decoded = layers.Dense(self.max_dimensions, activation='relu')(encoded)

        # this model maps an input to its reconstruction
        autoencoder = keras.Model([input_data,inputs_w], decoded)
        ae_optimizer = keras.optimizers.SGD(learning_rate = 0.01, clipnorm=0.1)
            
        autoencoder.compile(optimizer= ae_optimizer, loss=my_loss(inputs_w))
            
        self.explanation = autoencoder
            
            
    def update_explanation(self): # fitting autoencoder
        data_measured = np.array(self.data_measured) 
        data_to_predict = np.array(self.data_true)
        # input weights -- pass information on dimensions which the agent needed to predict
        input_weights = np.array(data_measured==-500).astype(int)
        
        # fitting until convergence
        self.explanation.fit([data_measured-100,input_weights], data_to_predict, 
            epochs=100000000,
            batch_size=1,
            shuffle=True,
            validation_data=([data_measured-100,input_weights], data_to_predict), verbose = False, callbacks=[overfitCallback])
            
    def evaluate_on_collected_data(self): # computing training error
        data_measured = np.array(self.data_measured)
        input_weights = np.array(data_measured==-500).astype(int)
        score = self.explanation.evaluate([data_measured-100,input_weights], np.array(self.data_true))
        return score
    
    def update_data(self, datapoint_measured, datapoint_true): # updating dataset with new data
        self.data_measured.append(datapoint_measured)
        self.data_true.append(datapoint_true)
        


# function to evaluate test performance
def evaluate_performance(scientist, environment, n=10000):
    score = None
    ground_truth_sample = []
    for i in range(n):
        obs = environment.sample()
        ground_truth_sample.append(obs)
    ground_truth_sample = np.array(ground_truth_sample)
    
    # MASKING RANDOM DIMENSIONS OF THE OBSERVATIONS
    measured_dims = np.random.rand(ground_truth_sample.shape[0], ground_truth_sample.shape[1]).argsort(axis=1)[:,0:scientist.measurement_capacity]
    measured_observations = np.zeros(ground_truth_sample.shape)-500
    measured_observations[np.expand_dims(np.arange(ground_truth_sample.shape[0]),1), measured_dims] = ground_truth_sample[np.expand_dims(np.arange(ground_truth_sample.shape[0]),1), measured_dims]
    
    input_weights = np.array(measured_observations==-500).astype(int)


    score = scientist.explanation.evaluate([measured_observations-100,input_weights], ground_truth_sample) # centering around~0
        
    return score

# SIMULATION CONDITIONS

# trial_index is used to pick conditions for the simulation run
    
# used the next line if you want to sweep through all conditions (I run them on a cluster)
# trial_index = int(sys.argv[1])

# use the following line to choose conditions manually
trial_index = 0


# setting up conditions for the current experiment
cond_list = []

# specifying parameters for the simulations
n_dimensions = [4,8,100] # dimensionality of the ground truth
dim_length = 200 # dimension length for the ground truth
n_clusters = [1, 10, 100] # n of clusters in the ground truth
measurement_capacities = ["n_dimensions/4", "n_dimensions/2"] # how many dimensions of an observation the agent observes (particular dimensions to mask/unmask will be chosen randomly for an observation)
samples = 4 # how many simulations will be run for each combination of conditions
explanation_capacities = [1,2,3,4,6,8,10,16,32,100,250,500,1000] # autoencoder width
wishart_scale = 5 # used to create the ground truth

for exp_cap in explanation_capacities:
    for dim in n_dimensions:
        for clst in n_clusters:
            for cap in measurement_capacities:
                for j in range(samples):
                    cond_list.append([cap, dim, clst, exp_cap])

# decoding conditions for a particular simulation        
env_dims = cond_list[trial_index][1]
measurement_capacity_cond = cond_list[trial_index][0]
if measurement_capacity_cond == "n_dimensions/2":
    measurement_capacity = int(env_dims//2)
elif measurement_capacity_cond == "n_dimensions/4":
    measurement_capacity = int(env_dims//4)

env_clusters = cond_list[trial_index][2]
explanation_capacity = cond_list[trial_index][3]

# useless parameters from previous simulations which were not varied but were saved nevertheless
# (their values don't mean anything for the current simulation) -- keeping them here just to keep consistency with the data files
explanation_strategy = "nn_autoencoder" 
exp_control_strategy = "close"
collective_strategy = "full data sharing"
strategy = "random"
n_scientists = 1
measurement_strategy = "safe"

parameters = {"n_scientists": n_scientists, "n_dimensions": env_dims,
             "dim_length": dim_length, "wishart_scale": wishart_scale, "n_clusters": env_clusters,
             "ag_max_dimension": env_dims, "measurement_capacity": measurement_capacity,
             "exp_control_strategy": exp_control_strategy, "experimentation_strategy": strategy,
             "measurement_strategy": measurement_strategy, "explanation_strategy": explanation_strategy,
             "explanation_capacity": explanation_capacity, "collective_strategy": collective_strategy}

print(parameters)


# SIMULATION STARTS HERE

local_performance = []
global_performance = []
  
# creating a multivariate gaussian environment
env = clustered_multivariate_gaussian(n_dims=env_dims, max_loc=dim_length, num_clusters=env_clusters, wishart_scale=wishart_scale)

# initializing the agent

agent = scientist(env_dims, measurement_capacity, explanation_capacity)
# initializing autoencoder
agent.initialize_explanation()

# evaluating agent's global performance before learning
global_performance.append([evaluate_performance(agent, env)])

# agent has 300 steps
for i in range(300):
    print(i)
    # collecting one new observationl; obs_measured -- masked observation; obs_true -- full observation    
    obs_measured, obs_true = agent.make_observation(env) 
    agent.update_explanation()
    
    # record the performance every 5 steps     
    if (i>0 and i%5 == 0):
        local_performance.append([agent.evaluate_on_collected_data()])       
        global_performance.append([evaluate_performance(agent, env)])
 

# saving the results    
d = [local_performance, global_performance, [], [], env]  
d.append(parameters)

# CHANGE TO YOUR DATA PATH -- n of the file will correspond to the number of observation
with open("YOUR_DATA_PATH/{}.pkl".format(trial_index), "wb") as fp:   #Pickling
    pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
