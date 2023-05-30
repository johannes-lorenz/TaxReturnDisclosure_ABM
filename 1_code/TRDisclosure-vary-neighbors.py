import matplotlib.pyplot as plt
import numpy as np
import time, enum, math
import pandas as pd
import pylab as plt
import networkx as nx
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
from tqdm import tqdm

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner, BatchRunnerMP


os.chdir('/Users/johannes/GitHub/Income_Tax_Return_Disclosure')

class Disclosure(Model):
    """A model with some number of agents."""
    def __init__(self, N = 100, scen = 3, audit_discl = 0, blockingfour = 1, kappa = 0.01, complexity = 2.09686, neighbors = 20, rho_upper = 0.1, intr_honest = 0.05, tax_rate = 0.35, initial_avoider = 0.5):
        self.num_nodes = N
        self.scenario = scen
        self.audit_disclosure = audit_discl
        self.blocking4 = blockingfour
        nearest_neighbors = neighbors
        rewiring_prob = kappa
        self.complexity = complexity
        self.taxrate = tax_rate
        self.intrinsically_honest = intr_honest
        self.socialpressure = rho_upper
        self.initial_avoider = initial_avoider
        self.G = nx.watts_strogatz_graph(n=self.num_nodes, k=nearest_neighbors, p=rewiring_prob)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.running = True

        """ Create agents """
        for i, node in enumerate(self.G.nodes()):
            a = TaxAgent(i + 1, self, self.taxrate, self.complexity, self.socialpressure, self.blocking4, self.audit_disclosure, self.scenario)
            self.schedule.add(a)
            """ Add the agent to a random grid cell"""
            self.grid.place_agent(a, node)
            
            """ make agents intrinsically honest, honest, avoiders or evaders at start """
            initial_state = np.random.choice([0,4], p=[self.intrinsically_honest, 1 - self.intrinsically_honest])
            if initial_state == 0:
                a.state = State.INTR_HONEST
                a.former_strategy = State.INTR_HONEST
            elif initial_state == 1:
                a.state = State.HONEST
                a.former_strategy = State.HONEST
            elif initial_state == 2:
                a.state = State.AVOIDER
                a.former_strategy = State.AVOIDER
            elif initial_state == 3:
                a.state = State.EVADER
                a.former_strategy = State.EVADER
            else:
                a.state = 4
                a.former_strategy = 4

        self.datacollector = DataCollector(
            model_reporters={"Number of Honest": agg_honest,
                             "Number of Avoiders": agg_avoiders,
                             "Number of Evaders": agg_evaders,
                             "TaxRev": agg_taxrev},
            agent_reporters={"State": "state",
                             "Number Neighbors": "number_neighbors",
                             "Honest Neighbors": "number_honest_neighbors",
                             "Avoiding Neighbors": "number_avoider_neighbors",
                             "Evading Neighbors": "number_evader_neighbors",
                             "Caught Evader Neighbors": "number_unlucky_evader_neighbors",
                             "Evasion-Amount": "opt_evasion",
                             "Evasion-Value": "value_evasion",
                             "Avoidance-Amount": "opt_avoidance",
                             "Avoidance-Value": "value_avoidance",
                             "Honesty-Value": "value_honesty",
                             "Audit-Counter": "audit_counter",
                             "Audit-Prob": "audit_prob",
                             "Erwischte": "caught",
                             "Optimization Cost": "optimization_cost",
                             "Tax payment": "tax"})


    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()



class TaxAgent(Agent):
    def __init__(self, unique_id, model, tax_rate, complexity, socialpressure, blocking4, audit_disclosure, scenario):
        super().__init__(unique_id, model)
        self.income = 1
        self.tax_rate = tax_rate
        self.penalty_rate = 1.6
        self.audit_prob = 0.05
        self.complexity = complexity
        self.risk_aversion = 10
        self.socialpressure = socialpressure
        
        self.optimization_cost = np.random.uniform(0.075,0.1)
        self.social_interaction_intensity = np.random.uniform(0,self.socialpressure)
        
        
        """ Settings """
        self.blocking4 = blocking4
        self.audit_disclosure = audit_disclosure
        self.scenario = scenario
        
        """ Initiate Variables for first round """
        self.number_neighbors = 0
        self.number_honest_neighbors = 0
        self.number_avoider_neighbors = 0
        self.number_evader_neighbors = 0
        self.number_unlucky_evader_neighbors = 0
        self.opt_evasion = 0
        self.value_evasion = 0
        self.opt_avoidance = 0
        self.value_avoidance = 0
        self.value_honesty = 0
        self.sum_honesty = 0
        self.sum_avoidance = 0
        self.sum_evasion = 0
        self.audit_counter = 0
        self.caught = 0
        self.tax = self.tax_rate * self.income#First period: No optimization possible
        if self.blocking4 == 1:
            self.abstinence_period = 5
        elif self.blocking4 == 0:
            self.abstinence_period = 1
        
        """ Give Agent a random strategy bias """
        self.honest_epsilon = np.random.normal(0, 0.05)
        self.avoider_epsilon = np.random.normal(0, 0.05)
        self.evader_epsilon = np.random.normal(0, 0.05)
    
    """ Ask about neighbors' current strategies """
    """ Unlucky evaders only count immediately after detection (i.e. not for four years) """
    def ask_neighbors(self):
        self.number_honest_neighbors = 0
        self.number_avoider_neighbors = 0
        self.number_evader_neighbors = 0
        self.number_unlucky_evader_neighbors = 0
        self.number_neighbors = 0
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        for a in self.model.grid.get_cell_list_contents(neighbors):
            self.number_neighbors += 1
            if a.caught == 1:
                self.number_unlucky_evader_neighbors += 1
            if a.state == State.AVOIDER:
                self.number_avoider_neighbors += 1
            if a.state == State.EVADER:
                self.number_evader_neighbors += 1
            if a.state == 0 or a.state == 1:
                self.number_honest_neighbors += 1
        
    
    """ Choose new strategy """
    def optimization(self):
        
        """ Estimate audit probability if model.audit_disclosure == 0 and replace audit_prob with this value """
        if self.audit_disclosure == 0 and self.state != 4:
            if self.scenario == 1:
                self.audit_prob = self.number_unlucky_evader_neighbors / (self.number_neighbors / 3)
            elif self.scenario == 2:
                if self.number_unlucky_evader_neighbors == 0:
                    self.audit_prob = 0
                else:
                    self.audit_prob = self.number_unlucky_evader_neighbors / ( ( self.number_neighbors - self.number_honest_neighbors ) / 2 )
            else:
                if self.number_unlucky_evader_neighbors == 0:
                    self.audit_prob = 0
                else:
                    self.audit_prob = self.number_unlucky_evader_neighbors / self.number_evader_neighbors
        
        """ Calculate PEU for avoidance, evasion, and honesty """
        if 0 < self.audit_prob < 1:
            self.opt_evasion = 1 / ( self.risk_aversion * self.income * self.penalty_rate ) * np.log( self.tax_rate * ( 1 - self.audit_prob ) / ( self.audit_prob * ( self.penalty_rate - self.tax_rate ) ) )
        elif self.audit_prob == 0:
            self.opt_evasion = 1
        elif self.audit_prob == 1:
            self.opt_evasion = 0
        if 0 < self.opt_evasion < 1:
            self.value_evasion = - (1 - self.audit_prob) * np.exp( - self.risk_aversion * self.income * ( 1 - self.tax_rate ) ) * self.penalty_rate / ( self.penalty_rate - self.tax_rate ) * ( (1 - self.audit_prob) * self.tax_rate / ( self.audit_prob * (self.penalty_rate - self.tax_rate) ) ) ** (- self.tax_rate / self.penalty_rate)
        elif self.opt_evasion <= 0:
            self.value_evasion = - np.exp( - self.risk_aversion * self.income * ( 1 - self.tax_rate ) )
        elif self.opt_evasion >= 1:
            self.value_evasion = - np.exp( - self.risk_aversion * self.income ) * ( 1 - self.audit_prob + self.audit_prob * np.exp( self.risk_aversion * self.income * self.penalty_rate ) )
        
        self.opt_avoidance = 1 / ( self.risk_aversion * self.income ) * ( 1 / self.optimization_cost - self.complexity / self.tax_rate )
        if self.opt_avoidance > 1:
            self.opt_avoidance = 1
        elif self.opt_avoidance < 0:
            self.opt_avoidance = 0
        self.value_avoidance = - self.optimization_cost * self.complexity / self.tax_rate * np.exp( 1 - self.risk_aversion * self.income * (1 - self.tax_rate) - self.optimization_cost * self.complexity / self.tax_rate )
        
        self.value_honesty = - np.exp( - self.risk_aversion * self.income * ( 1 - self.tax_rate ) )
        
        """ Calculate SU -- Scenario 2 """
        if self.scenario == 2:
            self.su_evasion = - np.exp( - self.social_interaction_intensity * ( - self.number_unlucky_evader_neighbors / self.number_neighbors - self.number_honest_neighbors / self.number_neighbors + self.evader_epsilon ) )
            self.su_avoidance = - np.exp( - self.social_interaction_intensity * ( - self.number_honest_neighbors / self.number_neighbors + self.avoider_epsilon ) )
            self.su_honesty = - np.exp( - self.social_interaction_intensity * ( self.number_honest_neighbors / self.number_neighbors - ( self.number_neighbors - self.number_honest_neighbors) / self.number_neighbors + self.honest_epsilon ) )
    
        """ Calculate SU -- Scenario 3 """
        if self.scenario == 3:
            self.su_evasion = - np.exp( - self.social_interaction_intensity * ( self.number_evader_neighbors / self.number_neighbors - self.number_unlucky_evader_neighbors / self.number_neighbors - self.number_avoider_neighbors / self.number_neighbors - self.number_honest_neighbors / self.number_neighbors + self.evader_epsilon ) )
            self.su_avoidance = - np.exp( - self.social_interaction_intensity * ( self.number_avoider_neighbors / self.number_neighbors - self.number_evader_neighbors / self.number_neighbors - self.number_honest_neighbors / self.number_neighbors + self.avoider_epsilon ) )
            self.su_honesty = - np.exp( - self.social_interaction_intensity * ( self.number_honest_neighbors / self.number_neighbors - self.number_avoider_neighbors / self.number_neighbors - self.number_evader_neighbors / self.number_neighbors + self.honest_epsilon ) )
            
        """ Calculate SU -- Scenario 1 """
        if self.scenario == 1:
            self.su_evasion = - np.exp( - self.social_interaction_intensity * ( 1/3 - 1/3 - 1/3 - self.number_unlucky_evader_neighbors / self.number_neighbors  + self.evader_epsilon ) )
            self.su_avoidance = - np.exp( - self.social_interaction_intensity * ( 1/3 - 1/3 - 1/3 + self.avoider_epsilon ) )
            self.su_honesty = - np.exp( - self.social_interaction_intensity * ( 1/3 - 1/3 - 1/3 + self.honest_epsilon ) )
        
        """ Sum of PEU and SU """
        self.sum_evasion = self.value_evasion + self.su_evasion
        self.sum_avoidance = self.value_avoidance + self.su_avoidance
        self.sum_honesty = self.value_honesty + self.su_honesty
        
        
        
        """ Choice first period: regard private expected utility only """
        if self.state == 4:
            if self.value_evasion > self.value_avoidance and self.value_evasion > self.value_honesty:
                self.state = 3
            elif self.value_avoidance > self.value_evasion and self.value_avoidance > self.value_honesty:
                self.state = 2
            else:
                self.state = 1
        
        """ Make a choice if not intrinsically honest """
        if self.state != 0 and self.former_strategy != 4:
            if self.sum_evasion > self.sum_avoidance and self.sum_evasion > self.sum_honesty and self.audit_counter == 0:
                self.state = 3
            elif self.sum_avoidance > self.sum_evasion and self.sum_avoidance > self.sum_honesty and self.audit_counter == 0:#Letzte Bedingung: erwischte Hinterzieher dürfen auch nicht optimieren.
                self.state = 2
            elif self.sum_honesty > self.sum_evasion and self.sum_honesty > self.sum_avoidance:
                self.state = 1
            elif self.audit_counter != 0:
                self.state = 1

    """ Audit tax evaders and collect tax payments from all taxpaers """
    def audit(self):
        self.former_strategy = self.state #Remember current strategy
        if self.state == State.EVADER:
            self.audit_try = np.random.uniform(0,1)
            if self.audit_try <= 0.05:
                self.caught = 1
                """ If blocking4 == 1: Set counter to 5 to stop evasion for four years (else: set to 1). Disounting starts immediately, that's why it's set to five """
                self.audit_counter = self.abstinence_period
                """ Calculate unlucky evader's tax payment """
                self.tax = self.tax_rate * self.income * (1 - self.opt_evasion) + self.penalty_rate * self.income * self.opt_evasion
            else:
                self.caught = 0
                """ Calculate lucky evaders' tax payment """
                self.tax = self.tax_rate * self.income * (1 - self.opt_evasion)
        else:
            self.caught = 0
            if self.state == State.INTR_HONEST or self.state == State.HONEST:
                """ Calculate honest taxpayer's tax payment """
                self.tax = self.tax_rate * self.income
            else:
                """ Calculate avoider's tax payment """
                self.tax = self.tax_rate * self.income * (1 - np.random.exponential(self.opt_avoidance / self.complexity))
    
    """ Discount audit blocking period """
    def discounting(self):
        if self.audit_counter > 0:
            self.audit_counter -= 1
    
    def step(self):
        self.ask_neighbors()
        self.optimization()
        self.audit()
        self.discounting()

class State(enum.IntEnum):
    INTR_HONEST = 0
    HONEST = 1
    AVOIDER = 2
    EVADER = 3



def agg_honest(model):
    number_honest = 0
    for agent in model.schedule.agents:
        if agent.state == 0 or agent.state == 1:
            number_honest += 1
    number_honest = number_honest / model.num_nodes
    return number_honest

def agg_avoiders(model):
    number_avoiders = 0
    for agent in model.schedule.agents:
        if agent.state == 2:
            number_avoiders += 1
    number_avoiders = number_avoiders / model.num_nodes
    return number_avoiders

def agg_evaders(model):
    number_evaders = 0
    for agent in model.schedule.agents:
        if agent.state == 3:
            number_evaders += 1
    number_evaders = number_evaders / model.num_nodes
    return number_evaders

def agg_taxrev(model):
    taxrev = 0
    for agent in model.schedule.agents:
        taxrev += agent.tax
    taxrev = taxrev / model.num_nodes #Needs to be adapted if income is not identical 1 for all agents
    return taxrev






#Scenario 1

number_iterations = 50

if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 1, "audit_discl": 0}
    variable_params = {"neighbors": range(2,20,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen1_1.pkl'))


if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 1, "audit_discl": 0}
    variable_params = {"neighbors": range(20,40,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen1_2.pkl'))


if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 1, "audit_discl": 0}
    variable_params = {"neighbors": range(40,60,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen1_3.pkl'))



if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 1, "audit_discl": 0}
    variable_params = {"neighbors": range(60,80,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen1_4.pkl'))




if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 1, "audit_discl": 0}
    variable_params = {"neighbors": range(80,101,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen1_5.pkl'))


# #Scenario 2

number_iterations = 50

if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 2, "audit_discl": 0}
    variable_params = {"neighbors": range(2,20,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen2_1.pkl'))




if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 2, "audit_discl": 0}
    variable_params = {"neighbors": range(20,40,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen2_2.pkl'))



if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 2, "audit_discl": 0}
    variable_params = {"neighbors": range(40,60,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen2_3.pkl'))



if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 2, "audit_discl": 0}
    variable_params = {"neighbors": range(60,80,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen2_4.pkl'))




if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 2, "audit_discl": 0}
    variable_params = {"neighbors": range(80,101,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen2_5.pkl'))





#Scenario 3

number_iterations = 50

if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 3, "audit_discl": 1}
    variable_params = {"neighbors": range(2,20,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen3_1.pkl'))




if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 3, "audit_discl": 1}
    variable_params = {"neighbors": range(20,40,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen3_2.pkl'))



if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 3, "audit_discl": 1}
    variable_params = {"neighbors": range(40,60,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen3_3.pkl'))



if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 3, "audit_discl": 1}
    variable_params = {"neighbors": range(60,80,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen3_4.pkl'))




if __name__ == '__main__':

    fixed_params = {"N": 1000, "blockingfour": 1, "intr_honest": 0.05, "scen": 3, "audit_discl": 1}
    variable_params = {"neighbors": range(80,101,1)}

    batch_run = BatchRunnerMP(
        model_cls=Disclosure,
        nr_processes=6,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=number_iterations,
        max_steps=41,
        model_reporters={"Number of Honest": agg_honest, "Number of Avoiders": agg_avoiders, "Number of Evaders": agg_evaders, "Tax": agg_taxrev},
        display_progress=True,
    )


    batch_run.run_all()

    run_modeldata_df = batch_run.get_model_vars_dataframe()

    run_modeldata_df.to_pickle(os.path.join('2_temp_files', 'modeldata_alt_neighbors_scen3_5.pkl'))
    