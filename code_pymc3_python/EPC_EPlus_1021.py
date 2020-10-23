#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Version : 3.7
# Author: Yaonan Gu
# Time:10/21/2020
# @File: 0505EPlus_EPC.py

import time
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib

import theano
import theano.tensor as tt
import pymc3 as pm
import scipy.stats as st
from scipy.stats import norm
import yaml
import time

from IPython.core.pylabtools import figsize
import os

from _10212020_EPC_EPlus import *

idx = pd.date_range('2018-01-01', periods=365, freq='D')
idx_H=pd.date_range('2018-01-01', periods=8760, freq='H')
idx_M = pd.date_range('2018-01-01', periods=12, freq='M')

conf = yaml.load(open("./simple_config.yaml"), Loader=yaml.FullLoader)
# assert conf["noise_ratio"] in [0.0, 0.1, 0.3, 0.5, 0.7]
with open('simple_config.yaml', 'w') as yaml_file:
    yaml_file.write( yaml.dump(conf, default_flow_style=False))

schedule_conf = yaml.load(open("./comb_schedules.yaml"), Loader=yaml.FullLoader)
with open('comb_schedules.yaml', 'w') as yaml_file:
    yaml_file.write( yaml.dump(schedule_conf, default_flow_style=False))

zone_par = yaml.load(open("./sm_zone_par.yaml"), Loader=yaml.FullLoader)
with open('sm_zone_par.yaml', 'w') as yaml_file:
    yaml_file.write( yaml.dump(zone_par, default_flow_style=False))

# assign weather file
wea_file= './Sim_Climate.csv'

from statistics import mode
import matplotlib.pyplot as plt

def call_EPC_2(conf, schedule_conf, zone_par, wea_file):

    ref_overall = pd.DataFrame(columns=['Etotal[kW/m2]'])

    ## Input weather data from csv
    weatherData, SRF_overhang, SRF_fin, SRF_horizon, Esol_30, Esol_45, Esol_60, Esol_90 = Climate(wea_file, "SRF.csv", "Esol.csv")

    ## Calculate building energy consumption
    # Ref, ref_deliveredEnergy, ref_Overall_deliveredEnergy = Hourly_EPC("RefBldgMediumOffice_Input_3.csv", weatherData, SRF_overhang, SRF_fin, SRF_horizon, Esol_30, Esol_45, Esol_60, Esol_90,values_list[0],values_list[1],values_list[2])

    ref, ref_deliveredEnergy, ref_Overall_deliveredEnergy,ref_deliveredEnergy_fuel, Fan_energy, cooling_energy, Pump_energy = Hourly_EPC("simple_model_0822.csv", weatherData, SRF_overhang, SRF_fin, SRF_horizon, Esol_30, Esol_45, Esol_60, Esol_90,conf, schedule_conf, zone_par)

#     ref_De_H=ref_De_H.append(pd.Series(ref_deliveredEnergy,index=['Etotal[kWh/m2]']),ignore_index=True)

    ref_De_H=pd.DataFrame({'Etotal[W/m2]':ref_deliveredEnergy[:,9]})
    ref_overall=ref_overall.append(pd.Series(ref_Overall_deliveredEnergy,index=['Etotal[kW/m2]']),ignore_index=True)
    ref_De_D=pd.DataFrame({'Etotal[W/m2]':np.add.reduceat(ref_deliveredEnergy[:,9],np.arange(0,len(ref_deliveredEnergy[:,9]),24))/24}, index=idx)
    ref_De_M=ref_De_D.resample('M').mean()
    # print(ref_De_M) #Check the running times
    return ref_deliveredEnergy, ref_Overall_deliveredEnergy,ref_deliveredEnergy_fuel, Fan_energy, cooling_energy, Pump_energy,ref_De_M,ref_De_D,ref_De_H, ref_overall

list_1 = ['Electricity:Facility ','Heating:Electricity ','Cooling:Electricity ','InteriorLights:Electricity ','Fans:Electricity ','Pumps:Electricity ','InteriorEquipment:Electricity ']
list_2 = ['Etotal[W/m2]','Eheat[W/m2]', 'Ecool[W/m2]', 'Elight[W/m2]', 'Efan[W/m2]', 'Epump[W/m2]','Equip[W/m2]']
list_3 = ['Etotal','Eheat', 'Ecool', 'Elight', 'Efan', 'Epump','Equip']


# define loglike class and function
class LogLike(tt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    # delete x in the required inputs
    def __init__(self, loglike, data, sigma, conf, schedule_conf, zone_par):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        #         self.x = x
        self.sigma = sigma
        self.conf = conf
        self.schedule_conf = schedule_conf
        self.zone_par = zone_par

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.data, self.sigma, self.conf, self.schedule_conf, self.zone_par)

        outputs[0][0] = np.array(logl)  # output the log-likelihood


def my_loglike(theta, data, sigma, conf, schedule_conf, zone_par):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """
    conf['Cooling_setpoint_weekday']['est'], conf['Cooling_setpoint_weekend']['est'] = theta
    print(conf['Cooling_setpoint_weekday']['est'])
    print(conf['Cooling_setpoint_weekend']['est'])
    #     schedule_conf = schedule_conf
    ref_deliveredEnergy, ref_Overall_deliveredEnergy, ref_deliveredEnergy_fuel, Fan_energy, cooling_energy, Pump_energy, ref_De_M, ref_De_D, ref_De_H, ref_overall = call_EPC_2(
        conf, schedule_conf, zone_par, wea_file)

    ref_all_H_idx = pd.DataFrame(ref_deliveredEnergy[:, 0:10],
                                 columns=['Eheat[W/m2]', 'Ecool[W/m2]', 'Elight[W/m2]', 'Efan[W/m2]', 'Epump[W/m2]',
                                          'Equip[W/m2]', 'Edhw[W/m2]', 'Egen_pv[W/m2]', 'Egen_wind[W/m2]',
                                          'Etotal[W/m2]'], index=idx_H)

    ref_all_D_idx = ref_all_H_idx.resample('D').mean()
    ref_all_M_idx = ref_all_H_idx.resample('M').mean()

    yest = ref_all_M_idx[list_2[3]]

    #     model = my_model(theta, x)

    return -(0.5 / sigma ** 2) * np.sum((data - yest) ** 2)

# core = 1, formal run with pymc3 model
t_0 = time.process_time()
# set up our data
N = 10  # number of data points
sigma = 1.  # standard deviation of noise

# TODO: Change x to weather data
# x = np.linspace(0., 9., N)

# mtrue = 0.4  # true gradient
# ctrue = 3.   # true y-intercept

# make observation data
df = pd.read_csv('./Simple_eplus_decomposition/Simple_eplus_decomposition_M.csv')
data = df[list_1[3]]

ndraws = 1000  # number of draws from the distribution
nburn = 200  # number of "burn-in points" (which we'll discard)

# create our Op
logl = LogLike(my_loglike, data, sigma, conf, schedule_conf, zone_par)

# use PyMC3 to sampler from log-likelihood
with pm.Model():
    # uniform priors on m and c
    #     m = pm.Uniform('m', lower=-10., upper=10.)
    #     c = pm.Uniform('c', lower=-10., upper=10.)
    t1 = pm.Uniform("t1", lower=conf['Cooling_setpoint_weekday']['lower'],
                    upper=conf['Cooling_setpoint_weekday']['upper'])

    t2 = pm.Uniform("t2", lower=conf['Cooling_setpoint_weekend']['lower'],
                    upper=conf['Cooling_setpoint_weekend']['upper'])

    #         conf['Heating_setpoint_weekday']['est'] = pm.Uniform("t2",lower=conf['Heating_setpoint_weekday']['lower'], upper=conf['Heating_setpoint_weekday']['upper']).random(size=1)
    #     conf['Cooling_setpoint_weekend']['est'] = pm.Uniform("t3",lower=conf['Cooling_setpoint_weekend']['lower'], upper=conf['Cooling_setpoint_weekend']['upper']).random(size=1)

    #     t1_print = tt.printing.Print('t1')(conf['Cooling_setpoint_weekday']['est'])
    #     t3_print = tt.printing.Print('t3')(conf['Cooling_setpoint_weekend']['est'])
    #     t7_print = tt.printing.Print('t7')(t7)

    #     for i in range(4, 7):
    #         globals()["t%s" % i] = pm.Uniform("t{}".format(i),lower=0.8, upper=1.2).random(size=1)
    #     t4_print = tt.printing.Print('t4')(t4)
    #     t5_print = tt.printing.Print('t5')(t5)
    #     t6_print = tt.printing.Print('t6')(t6)

    #     for s in range(1, len(zone_par.keys())+1):
    #         zone_par["Zone{}".format(s)]['Appliance (W/m2)'] = zone_par["Zone{}".format(s)]['Appliance (W/m2)']*t4
    #         zone_par["Zone{}".format(s)]['Occupancy (m2/person)'] = zone_par["Zone{}".format(s)]['Occupancy (m2/person)']*t5
    #         zone_par["Zone{}".format(s)]['Lighting (W/m2)'] = zone_par["Zone{}".format(s)]['Lighting (W/m2)']*t6

    # convert m and c to a tensor vector
    theta = tt.as_tensor_variable([t1, t2])

    # use a DensityDist (use a lamdba function to "call" the Op)
    pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})

    # Using NUTS sampling
    #     step_name = pm.NUTS()

    #     db=pm.backends.Text('./Simple_Model/Hourly')

    # Sample from the posterior
    #     trace = pm.sample(2000, tune=1000,step=step_name, cores=2,trace=db)
    #     trace = pm.sample(1000, tune=500,step=step_name, cores=2)

    trace = pm.sample(ndraws, tune=nburn, chains=2, cores=1, discard_tuned_samples=False)

# # plot the traces
# _ = pm.traceplot(trace, lines={'t1_true': 23.9, 't2_true': 23.9})

# # put the chains in an array (for later!)
# samples_pymc3 = np.vstack((trace['t1'], trace['t2'])).T


elapsed_time = time.process_time() - t_0
print('The total execution time of running Ref is {} seconds'.format(elapsed_time))

pm.traceplot(trace)
plt.show()
pm.summary(trace)