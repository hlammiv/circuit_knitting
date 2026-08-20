#!/usr/bin/env python
# coding: utf-8

# In[2]:


import import_ipynb
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_ibm_runtime import SamplerV2, Batch
from qiskit_aer.primitives import EstimatorV2

from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

from functions import *
from improved.config import ExperimentConfig as BatchedExperimentConfig
from improved.experiment import circuit_knitter as batched_circuit_knitter


def circuit_knitter(
    qc, conq, tarq, num_shots, simulator_seed=1, transpiler_seed=1, noise=False
):
    """Compatibility wrapper routing production runs through the batched engine."""
    config = BatchedExperimentConfig(noise=noise, num_shots=num_shots)
    result = batched_circuit_knitter(
        qc,
        conq,
        tarq,
        num_shots,
        config,
        simulator_seed=simulator_seed,
        # Compilation is intentionally fixed across statistical replicas.
        transpiler_seed=20260819,
    )
    return result["results"]


from params import *


# In[4]:


# """ step 1, no noise, knitted, convergences as function of shots """
# np.random.seed(1)
# for i in range(11):
#     num_shots = 1024*2**i
#     with open('data/step1_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_no_noise_knitted.pkl', 'wb') as file:
#         pickle.dump(circuit_knitter(trot_step_1, 0, 10, num_shots, simulator_seed=np.random.randint(1024**2),\
#                                                      transpiler_seed=np.random.randint(1024**2)),\
#                     file, protocol=pickle.HIGHEST_PROTOCOL)


# In[5]:


# """ step 1, noisy, knitted, convergence as function of shots """
# np.random.seed(2)
# for i in range(11):
#     num_shots = 1024*2**i
#     with open('data/step1_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_with_noise_knitted.pkl', 'wb') as file:
#         pickle.dump(circuit_knitter(trot_step_1, 0, 10, num_shots, noise=True, simulator_seed=np.random.randint(1024**2),\
#                                                      transpiler_seed=np.random.randint(1024**2)),\
#                     file, protocol=pickle.HIGHEST_PROTOCOL)


# In[6]:


# """ step 2, no noise, knitted, convergences as function of shots """
# np.random.seed(3)
# for i in range(11):
#     num_shots = 1024*2**i
#     with open('data/step2_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_no_noise_knitted.pkl', 'wb') as file:
#         pickle.dump(circuit_knitter(trot_step_2, 0, 10, num_shots, simulator_seed=np.random.randint(1024**2),\
#                                                      transpiler_seed=np.random.randint(1024**2)),\
#                     file, protocol=pickle.HIGHEST_PROTOCOL)


# In[7]:


# """ step 2, noisy, knitted, convergence as function of shots """
# np.random.seed(4)
# for i in range(11):
#     num_shots = 1024*2**i
#     with open('data/step2_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_with_noise_knitted.pkl', 'wb') as file:
#         pickle.dump(circuit_knitter(trot_step_2, 0, 10, num_shots, noise=True, simulator_seed=np.random.randint(1024**2),\
#                                                      transpiler_seed=np.random.randint(1024**2)),\
#                     file, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


# """ step 1, noisy, knitted, convergence as function of epsilon (epsilon = 0.2, 0.4, 0.6, 0.8)"""
# np.random.seed(5)
# for i in range(4):
#     num_shots = 131072
#     epsilon = np.round(0.2*(1+i), 1)
#     circuit = trotter_stepper(1, Nqbits, epsilon, mass, mid).decompose().decompose()
#     circuit.measure_all()
#     with open('data/step1_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_with_noise_knitted.pkl', 'wb') as file:
#          pickle.dump(circuit_knitter(circuit, 0, 10, num_shots, noise=True, simulator_seed=np.random.randint(1024**2),\
#                                                      transpiler_seed=np.random.randint(1024**2)),\
#                      file, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


# """ step 2, noisy, knitted, convergence as function of epsilon (epsilon = 0.5, 0.6, 0.7, 0.8)"""
# np.random.seed(6)
# for i in range(4):
#     num_shots = 65536
#     epsilon = np.round((5+i)*0.1, 1)
#     circuit = trotter_stepper(2, Nqbits, epsilon, mass, mid).decompose().decompose()
#     circuit.measure_all()
#     with open('data/step2_epsilon' + str(epsilon)[0] + str(epsilon)[2] + '_count' + str(num_shots) + '_with_noise_knitted.pkl', 'wb') as file:
#          pickle.dump(circuit_knitter(circuit, 0, 10, num_shots, noise=True, simulator_seed=np.random.randint(1024**2),\
#                                                      transpiler_seed=np.random.randint(1024**2)),\
#                      file, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


seeds_list = []
with open('data/noisy_knitted_trotter_evol/noisy_knitted_seeds_list.txt', 'r') as file:
    for line in file:
        val = line[:-1]
        seeds_list.append(int(val))
my_seed=seeds_list[0]
np.random.seed(my_seed)
for i in range(4):
    num_shots = 16384
    epsilon = np.round(0.2*(1+i), 1)
    circuit = trotter_stepper(1, Nqbits, epsilon, mass, mid).decompose().decompose()
    circuit.measure_all()
    res = circuit_knitter(circuit, 0, 10, num_shots, noise=True, simulator_seed=np.random.randint(1024**2),\
                                                     transpiler_seed=np.random.randint(1024**2))
    with open('data/noisy_knitted_trotter_evol/TEST_step1_epsilon' + str(epsilon)[0] + str(epsilon)[2] +\
              '_count' + str(num_shots) + '_with_noise_knitted_seed' + str(my_seed) + '.pkl', 'wb') as file:
         pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


for _ in range(8):
    seeds_list = []
    try:
        with open('data/noisy_knitted_trotter_evol/noisy_knitted_seeds_list.txt', 'r') as file:
            for line in file:
                val = line[:-1]
                seeds_list.append(int(val))
    except FileNotFoundError:
        pass
    my_seed = int(datetime.now().timestamp())
    seeds_list.append(my_seed)
    np.random.seed(my_seed)
    for i in range(4):
        num_shots = 16384
        epsilon = np.round(0.2*(1+i), 1)
        circuit = trotter_stepper(1, Nqbits, epsilon, mass, mid).decompose().decompose()
        circuit.measure_all()
        res = circuit_knitter(circuit, 0, 10, num_shots, noise=True, simulator_seed=np.random.randint(1024**2),\
                                                         transpiler_seed=np.random.randint(1024**2))
        with open('data/noisy_knitted_trotter_evol/step1_epsilon' + str(epsilon)[0] + str(epsilon)[2] +\
                  '_count' + str(num_shots) + '_with_noise_knitted_seed' + str(my_seed) + '.pkl', 'wb') as file:
             pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)
    for i in range(4):
        num_shots = 16384
        epsilon = np.round((5+i)*0.1, 1)
        circuit = trotter_stepper(2, Nqbits, epsilon, mass, mid).decompose().decompose()
        circuit.measure_all()
        res = circuit_knitter(circuit, 0, 10, num_shots, noise=True, simulator_seed=np.random.randint(1024**2),\
                                                         transpiler_seed=np.random.randint(1024**2))
        with open('data/noisy_knitted_trotter_evol/step2_epsilon' + str(epsilon)[0] + str(epsilon)[2] +\
                  '_count' + str(num_shots) + '_with_noise_knitted_seed' + str(my_seed) + '.pkl', 'wb') as file:
             pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/noisy_knitted_trotter_evol/noisy_knitted_seeds_list.txt', 'w') as file:
        for item in seeds_list:
            file.write("%s\n" % item)

