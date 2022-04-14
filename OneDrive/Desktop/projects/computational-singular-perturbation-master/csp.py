
# Computation singular peturbation method implemented in 
# Python programming langauge by Dr. Harish Kumar Chakravarty and Prof. Alexis Matynia, Prof. Patrick De Costa, and Prof. Anca Belme at 
# Jean Le Rond d'Alembert Institute - Sorbonne University www.dalembert.upmc.fr

#!/usr/bin/env python
# coding: utf-8
# Read the Parameters, Frame Files

import configparser
import time

# importing all the required libraries........................................

import warnings
warnings.filterwarnings('ignore')
import cantera as ct
import numpy as np
import scipy.integrate
import sympy as sym
import matplotlib.pyplot as plt

# import definitions
# get_ipython().run_line_magic('matplotlib', 'inline')

start = time.time()
config = configparser.ConfigParser()

# User must enter the name of input file here (such as param_general.ini)

config.read('param_general.ini') # enter path of the parameter file

filename = config.get('parameters','filename')
t_end = config.getfloat('parameters','end_time')
T = config.getfloat('parameters','T')
P = config.getfloat('parameters','P')*ct.one_atm
X = config.get('parameters','X')

equiv_ratio = config.getfloat('parameters','equiv_ratio')
fuel = config.get('parameters','fuel')
oxidizer = config.get('parameters','oxidizer')
inert_gas = config.get('parameters','inert_gas')
kernel = config.get('parameters','kernel').split()


# Refinement functions: The basis vectors generated from the Jacobian matrix provide 
# leading order approximation and has been used as intial trial basis vectors in the refinement procedure. 
# Since basis vectors changes as a function of time, it is important to perform two step
# CSP refinement procedure to generate the correct refined set of vectors valid at all the time (time-dependent basis vector).
# In general, CSP method provides programmable two-step recursive procedure for refinement of the linearly  independent basis vectors 
# obtained from the Jacobian matrix of chemical kinetic mechanism to generate the correct refined
# set of orthogonal time-dependent basis vector to develop a reduce model with higher accuracy
# Combustion Science and Technology, 89, 5-6, p. 375, 1993. 

# Defining the CSP refinement function

def refinement(a_initial, dadt, dbdt, nFastMode, conc):
    
    b_initial = np.linalg.inv(a_initial)
    
    dicts_conc = dict(zip(species_list,conc))
    J_val = np.array(J_fun.subs(dicts_conc),dtype=float)
    
    Lambda_initial = np.zeros((nFastMode,nFastMode),dtype=float)  # equation 6.15, without the derivative term.....
    for i in range(nFastMode):
        for j in range(nFastMode):
            Lambda_initial[i,j] = np.matmul(np.matmul(b_initial[i,:],J_val),a_initial[:,j])
    print('Initial Lambda matrix: {}\n'.format(Lambda_initial))
    tau_initial = np.linalg.inv(Lambda_initial)
    print('Initial Tau matrix: {}\n'.format(tau_initial))

    b_refined_1 = b_initial    
    for i in range(nFastMode):
        sum_ = np.zeros((1,nSpecies),dtype=float)
        for j in range(nFastMode):
            sum_ += np.real(tau_initial[i,j]*np.matmul(np.reshape(b_initial[j,:],(1,nSpecies)),J_val))
            b_refined_1[i,:] = sum_
    print('Rows of refined vector b (1), corresponding to fast modes (only): \n{}\n'.format(b_refined_1))

    Lambda_1 = np.zeros((nFastMode,nFastMode),dtype=float)  # equation 6.15, without the derivative term.....
    for i in range(nFastMode):
        for j in range(nFastMode):
            Lambda_1[i,j] = np.matmul(np.matmul(b_refined_1[i,:],J_val),a_initial[:,j])
    print('Initial Lambda matrix: {}\n'.format(Lambda_1))
    tau_1 = np.linalg.pinv(Lambda_1)
    print('Initial Tau matrix: {}\n'.format(tau_1))

    a_refined_1 = a_initial      
    for i in range(nFastMode):
        sum_ = np.zeros((nSpecies,1),dtype=float)
        for j in range(nFastMode):
            sum_ += np.real(tau_1[j,i]*np.matmul(J_val,np.reshape(a_initial[:,j],(nSpecies,1))))
            a_refined_1[:,i] = np.reshape(sum_,(nSpecies,))
    print('Rows of refined vector a (1), corresponding to fast modes (only): \n{}\n'.format(a_refined_1))

# step2: including the derivative term
    b_refined_2 = b_refined_1    
    for i in range(nFastMode):
        sum_ = np.zeros((1,nSpecies),dtype=float)
        for j in range(nFastMode):
            sum_ += np.real(tau_1[i,j]*(np.reshape(dbdt[:,j],(1,nSpecies))+np.matmul(np.reshape(b_refined_1[j,:],(1,nSpecies)),J_val)))
            b_refined_2[i,:] = sum_
    print('Rows of refined vector b (1), corresponding to fast modes (only): \n{}\n'.format(b_refined_1))

    Lambda_2 = np.zeros((nFastMode,nFastMode),dtype=float)  # equation 6.15, without the derivative term.....
    for i in range(nFastMode):
        for j in range(nFastMode):
            Lambda_2[i,j] = np.matmul(np.matmul(b_refined_2[i,:],J_val),a_refined_1[:,j])
    print('Initial Lambda matrix: {}\n'.format(Lambda_1))
    tau_2 = np.linalg.pinv(Lambda_2)
    print('Initial Tau matrix: {}\n'.format(tau_1))
    
    a_refined_2 = a_refined_1      
    for i in range(nFastMode):
        sum_ = np.zeros((nSpecies,1),dtype=float)
        for j in range(nFastMode):
            sum_ += np.real(tau_1[j,i]*(-np.reshape(dadt[:,j],(nSpecies,1))+np.matmul(J_val,np.reshape(a_refined_1[:,j],(nSpecies,1)))))
            a_refined_2[:,i] = np.reshape(sum_,(nSpecies,))
            

    print('Rows of refined vector a (1), corresponding to fast modes (only): \n{}\n'.format(a_refined_1))

    return sym.re(sym.Matrix(a_refined_2)), sym.re(sym.Matrix(b_refined_2)), tau_2


# Combustion and Flame 146 (2006) 29–51
def importanceIndex_Valorani(a_refined, b_refined, nFastMode, conc):
    
    dict_ = dict(zip(species_list,conc))
    R = F.subs(dict_)

    B = np.zeros((nSpecies,nReaction),dtype=float)
    for i in range(nSpecies):
        for j in range(nReaction):
            B[i,j] = np.array((b_refined[i,:]*Sr[:,j]).subs(dict_),dtype=float)[0,0]

    I_fast = np.zeros((nSpecies,nReaction),dtype=float)
    I_slow = np.zeros((nSpecies,nReaction),dtype=float)
    den = np.zeros((nSpecies,nReaction),dtype=float)

    for i in range(nSpecies):
        for j in range(nSpecies):
            for r in range(nFastMode):
                den[i,j] += float(a_refined[i,r])*B[r,j]*R[j]   
    den_ = np.sum(np.absolute(den),axis=1)


    for i in range(nSpecies):
        for k in range(nReaction):
            num  = 0
            for r in range(nFastMode):
                num += float(a_refined[i,r])*B[r,k]*R[k]
            I_fast[i,k] = num/den_[i]


    den = np.zeros((nSpecies,nSpecies),dtype=float)

    for i in range(nSpecies):
        for j in range(nSpecies):
            for r in range(nFastMode,nSpecies):
                den[i,j] += float(a_refined[i,r])*B[r,j]*R[j]   
    den_ = np.sum(np.absolute(den),axis=1)

    for i in range(nSpecies):
        for k in range(nReaction):
            num  = 0
            for r in range(nFastMode,nSpecies):
                num += float(a_refined[i,r])*B[r,k]*R[k]
            I_slow[i,k] = num/den_[i]
     
    return I_fast, I_slow

def participationIndex_Goussis(b_refined, nFastMode, conc):
    
    b = np.array(b_refined,dtype=float)
    P = np.zeros((nFastMode,nReaction),dtype=float)
    dict_ = dict(zip(species_list,conc))

    sum_ = np.zeros((nFastMode,1))
    for m in range(nFastMode):
        for k in range(nReaction):
            sum_[m,0] +=np.absolute(float(np.matmul(np.reshape(b[m,:],(1,nSpecies)),(np.array(Sr[:,k],dtype=float)*float(F[k].subs(dict_))))))
        
    for m in range(nFastMode):
        for j in range(nReaction):
            P[m,j] = float(np.matmul(np.reshape(b[m,:],(1,nSpecies)),(np.array(Sr[:,j],dtype=float)*float(F[j].subs(dict_)))))/sum_[m,0]

    return P


def importanceIndex_Goussis(a_refined, b_refined, nFastMode, conc):
   
    I = np.zeros((nSpecies,nReaction),dtype=float)
    dict_ = dict(zip(species_list,conc))
    
    s = nFastMode
    a = np.zeros((nSpecies,nSpecies))
    a[s:,s:] = np.array(a_refined[s:,s:],dtype=float)
    b = np.zeros((nSpecies,nSpecies))
    b[s:,s:] = np.array(b_refined[s:,s:],dtype=float)
#     A = np.matmul(a,b)
    A = np.matmul(np.array(a_refined,dtype=float), np.array(b_refined,dtype=float))

    sum_ = np.zeros((nFastMode,1))
    for n in range(nFastMode):
        for j in range(nReaction):
            sum_[n,0] +=np.absolute(float(np.matmul(np.reshape(A[n,:],(1,nSpecies)),(np.array(Sr[:,j],dtype=float)*float(F[j].subs(dict_))))))
        
    for n in range(nFastMode):
        for j in range(nReaction):
            I[n,j] = float(np.matmul(np.reshape(A[n,:],(1,nSpecies)),(np.array(Sr[:,j],dtype=float)*float(F[j].subs(dict_)))))/sum_[n,0]
            
    return I    


def projectionMatrix(a_refined, b_refined, nFastMode, conc):

    dicts_conc = dict(zip(species_list,conc))
    Q = np.zeros((nSpecies,nSpecies,nFastMode),dtype=float)
    radicals = []
    radIdx = []
    
    for i in range(nFastMode):
        
        Q[:,:,i] = np.array((a_refined[:,i]*b_refined[i,:]).subs(dicts_conc),dtype=float)

        idx = np.where(abs(np.diagonal(Q[:,:,i])-1)==min(abs(np.diagonal(Q[:,:,i])-1)))[0][0]
       
        radicals.append((i+1,idx))
        radIdx.append(idx)

    g_fast = sym.Matrix(np.sum(Q,axis=2))*g  
    g_slow = sym.Matrix(np.eye(nSpecies)-np.sum(Q,axis=2))*g
    
    radicalSpecies_list = [species_list[i] for i in radIdx]
    nonRadicalSpecies_list = [species for species in species_list if species not in radicalSpecies_list]

    return Q, radicalSpecies_list, nonRadicalSpecies_list, radIdx
   

def simplificationAlgorithm(nTime, kernel, tol):
    
    list_reaction = [str(reaction) for reaction in gas.reactions()]

    R_set = set()
    S_set = set(kernel)
    print('Tolerance: {} and kernel species: {}\n'.format(tol,kernel))
    S = set()
    S_global = set()
    R_global = set()
    for iTime in range(1,nTime-1):

        while (not(S==S_set)):
            S = S_set
            for species in S_set:
                for reaction in list_reaction:
                    if ((species in reaction.split()) and (I_slow[iTime][species_names.index(species),list_reaction.index(reaction)]>tol) and ((I_fast[iTime][radIdx[iTime],list_reaction.index(reaction)]>tol).all())):
                        print('\nIncluding reaction {}, as it contains species {}, \nwith I_slow: {} and I_fast: {}'.format(reaction,species,I_slow[iTime][species_names.index(species),list_reaction.index(reaction)],I_fast[iTime][radIdx[iTime],list_reaction.index(reaction)]))
                        R_set.add(str(reaction))

            for reaction in R_set:
                for species in species_names:
                    if (species in reaction.split() and not(float(Sr[species_names.index(species),list_reaction.index(reaction)])==0)):
                        print('\nIncluding species {}, as it appears in reaction {}.'.format(species, str(reaction)))
                        S_set.add(species)

            S_global = S_global.union(S_set)

            for species in S_set:
                for iReaction in range(nReaction):
                    reaction_list = list_reaction[iReaction].split()
                    reaction_list[:] = (value for value in reaction_list if value != '+')
                    if (species in reaction_list):
                        R_set.add(list_reaction[iReaction])
            R_global = R_global.union(R_set)
    print('\n\n Active species set: {}'.format(S_global))
    print('\n\nActive reaction set (#{}): \n{}'.format(len(R_global),R_global))
    print('----------------------')
    
    return S_global, R_global

def simplificationAlgorithmGoussis(nTime, kernel, tol):
    
    list_reaction = [str(reaction) for reaction in gas.reactions()]

    R_set = set()
    S_set = set(kernel)
    print('Tolerance: {} and kernel species: {}\n'.format(tol,kernel))
    S = set()
    S_global = set()
    R_global = set()
    for iTime in range(1,nTime-1):

        while (not(S==S_set)):
            S = S_set
            for species in S_set:
                for reaction in list_reaction:
                    if (species in reaction and np.absolute(I[iTime][species_names.index(species),list_reaction.index(reaction)])>tol and (np.absolute(I[iTime][species_names.index(species),list_reaction.index(reaction)])>tol).all):
                        print('\nIncluding reaction {}, as it contains species {}, \nwith |I|: {} and |I_radical|: {}'.format(reaction,species,np.absolute(I[iTime][species_names.index(species),list_reaction.index(reaction)]),np.absolute(I[iTime][species_names.index(species),list_reaction.index(reaction)])))
                        R_set.add(str(reaction))

            for reaction in R_set:
                for species in species_names:
                    if (species in reaction.split() and not(float(Sr[species_names.index(species),list_reaction.index(reaction)])==0)):
                        print('\nIncluding species {}, as it appears in reaction {}.'.format(species, str(reaction)))
                        S_set.add(species)

            S_global = S_global.union(S_set)

            for species in S_set:
                for iReaction in range(nReaction):
                    reaction_list = list_reaction[iReaction].split()
                    reaction_list[:] = (value for value in reaction_list if value != '+')
                    if (species in reaction_list):
                        R_set.add(list_reaction[iReaction])
            R_global = R_global.union(R_set)
    print('\n\n Active species set: {}'.format(S_global))
    print('\n\nActive reaction set (#{}): \n{}'.format(len(R_global),R_global))
    print('----------------------')
    
    return S_global, R_global

#..........................End of functins........................................................


# custom.py from cantera has been used only to get concentration of species as a function of time
# and solving a ignitions problem at constant pressure where the governing equations are implemented in Python


gas = ct.Solution(filename)
gas.TPX = T, P, X
# other conditions
y0 = np.hstack((gas.T, gas.Y))

gas.set_equivalence_ratio(equiv_ratio, fuel, oxidizer+','+inert_gas)

inlet = ct.Reservoir(gas)

class ReactorOde:
    def __init__(self, gas):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.P = gas.P

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        self.gas.set_unnormalized_mass_fractions(y[1:])
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp))
        dYdt = wdot * self.gas.molecular_weights / rho

        return np.hstack((dTdt, dYdt))


# Set up objects representing the ODE and the solver
ode = ReactorOde(gas)
solver = scipy.integrate.ode(ode)
solver.set_integrator('vode', method='bdf', with_jacobian=True)
solver.set_initial_value(y0, 0.0)

# Integrate the equations, keeping T(t) and Y(k,t)
# t_end = 1e-3
states = ct.SolutionArray(gas, 1, extra={'t': [0.0]})
dt = 1e-5
while solver.successful() and solver.t < t_end:
    solver.integrate(solver.t + dt)
    gas.TPY = solver.y[0], P, solver.y[1:]
    states.append(gas.state, t=solver.t)

#.........................end of custom.py..............................................................................................

# Evaluation of the global reaction rate vector using the stoichiometric vector and reaction rates.

species_names = gas.species_names
nSpecies, nReaction = len(species_names), len(gas.reactions())
print('No. of species: {}'.format(nSpecies))
print('Total no. of chemical reactions: {}'.format(nReaction))

from sympy.physics.vector import dynamicsymbols
species_list = dynamicsymbols(species_names)
sym.var('t');
print('Representation of species concentration (in symbols):\n{}'.format(species_list))

dicts_conc_initial = dict(zip(species_list,states.Y[0,:]))
print('Initial concentration: {}'.format(dicts_conc_initial))

list_ = []
for i in range(nSpecies):
    list_.append([0]*nReaction)
S_reactant = sym.Matrix(list_)
S_product = sym.Matrix(list_)
S_product.shape

list_ = []
for i in range(nSpecies):
    list_.append([0]*nReaction)
S_reactant = sym.Matrix(list_)
S_product = sym.Matrix(list_)
for i in range(nSpecies):
    for j in range(nReaction):
        S_reactant[i,j] = gas.reactant_stoich_coeffs()[i,j]
        S_product[i,j] = gas.product_stoich_coeffs()[i,j]
Sr = S_product - S_reactant
print('Stoichoiometric vectors: {}'.format(Sr.shape))

k_f = gas.forward_rate_constants
k_r = gas.reverse_rate_constants
print('rateconstants: {}'.format(k_f.shape))

F = sym.Matrix([0]*nReaction)
F_forward = sym.Matrix([0]*nReaction)
F_backward = sym.Matrix([0]*nReaction)
for iReaction in range(nReaction):
    reactant_species_list = gas.reactants(iReaction).split()
    reactant_species_list[:] = (value for value in reactant_species_list if value != '+')
    
    for species in reactant_species_list:
        if species.isdigit():
            idx = reactant_species_list.index(species)
            reactant_species_list[idx+1] = species + reactant_species_list[idx+1]
            reactant_species_list.remove(species)
    
    product_species_list = gas.products(iReaction).split()
    product_species_list[:] = (value for value in product_species_list if value != '+')
    
    for species in product_species_list:
        if species.isdigit():
            idx = product_species_list.index(species)
            product_species_list[idx+1] = species + product_species_list[idx+1]
            product_species_list.remove(species)
    
    factor = 1
    for species in reactant_species_list:
        if species[0].isdigit():
            factor*= int(species[0])
            idx = reactant_species_list.index(species)
            reactant_species_list[idx] = reactant_species_list[idx].replace(species[0],'',1)        
    product_of_reactants = factor*dynamicsymbols(reactant_species_list[0])
    for i in range(1,len(reactant_species_list)):
        product_of_reactants*=dynamicsymbols(reactant_species_list[i])

    factor = 1
    for species in product_species_list:
        if species[0].isdigit():
            factor*= int(species[0])
            idx = product_species_list.index(species)
            product_species_list[idx] = product_species_list[idx].replace(species[0],'',1)

    product_of_products = factor*dynamicsymbols(product_species_list[0])
    for i in range(1,len(product_species_list)):
        product_of_products*=dynamicsymbols(product_species_list[i])
    F_forward[iReaction] = k_f[iReaction]*product_of_reactants
    F_backward[iReaction] = k_r[iReaction]*product_of_products 
    F[iReaction]=(F_forward[iReaction]-F_backward[iReaction])
del reactant_species_list, product_species_list, product_of_reactants, product_of_products


# 1- Only for pressure-dependant reaction with collision partner
    #    => Calculation of M           -> omega = [M]*(...)
    
third_body_species_name = ['M', '(+M)']
third_body_species_list = dynamicsymbols(third_body_species_name)

import Class_def as cdef
mech_data = cdef.Mech_data(filename)
third_body_list = np.ones(len(mech_data.react.type)) #value of M.....
ns = gas.n_species

for r in range(len(mech_data.react.type)):  # loop on all reactions

    
    if mech_data.react.type[r] == 'three_body_reaction'    or mech_data.react.type[r] == 'falloff_reaction'    or mech_data.react.type[r] == 'pdep_arrhenius':

        # if there is no efficiency specified for the reaction
        if mech_data.react.eff[r]=='':
            third_body_list[r] = gas.P/(8.314*gas.T)

        # if there is efficiency specified for the reaction
        else:
            third_body_list[r]=0
                    # identification of collisional species
            col_sp = [] ; col_coeff = []
            for col_ in mech_data.react.eff[r]:
                for sp in range(ns):
                    if col_.split(':')[0] == gas.species_name(sp):
                        col_sp.append(sp)
                        col_coeff.append(float(col_.split(':')[1]))
                    # calculation of M as M = Sum(col_coeff*C_sp)
            for sp in range(ns):
                if sp in col_sp:
                    third_body_list[r] += col_coeff[col_sp.index(sp)]*gas.concentrations[sp]
                else:
                    third_body_list[r] += 1*gas.concentrations[sp]
                    
M = third_body_list
for i in range(nReaction):
    F[i] = F[i].subs({third_body_species_list[0]:M[i], third_body_species_list[1]:M[i]})
    F_forward[i] = F_forward[i].subs({third_body_species_list[0]:M[i], third_body_species_list[1]:M[i]})
    F_backward[i] = F_backward[i].subs({third_body_species_list[0]:M[i], third_body_species_list[1]:M[i]})

    
g=Sr*F
print('physical global reaction rate vector: {}'.format(g))

dicts_g = {}
for i in range(len(species_list)):
    dicts_g[i] = g[i].subs(dicts_conc_initial)
    dicts_g[species_list[i].diff()]=dicts_g.pop(i)

#...........................End of evaluation of global reaction rate vector........................

# Evaluation of jacobian matrix by differentiating g with respect to y.............................

nTime = len(states.Y)
conc = states.Y
# Jacobian
J_fun = sym.Matrix(nSpecies,nSpecies,[sym.diff(g[i],species_list[j]) for i in range(nSpecies) for j in range(nSpecies)])
print ('J_fun: \n{}\n'.format(J_fun))

J_val = []
for iTime in range(nTime):
    dicts_conc = dict(zip(species_list,conc[iTime,:]))
    J_val.append(np.array(J_fun.subs(dicts_conc),dtype=float))
print('Jacobian: \n{}\n'.format(J_val))

Tau = []# list containing the timeSceles at all times
for iTime in range(nTime):
    timeScales = np.reciprocal(np.sort(np.absolute(np.linalg.eigvals(J_val[iTime])))[::-1])
    Tau.append(timeScales)


# End of evalaution of Jacobian.................................................................

# Start of Decompostion of phase space.................................
for iTime in range(nTime):
    plt.semilogy(range(nSpecies),Tau[iTime],'.b')
plt.grid()
plt.ylabel('time (s)')
plt.xlabel('# modes ')    
plt.show()
TimeResolution = float(input('Enter a time resolution; reactions with time scales less than this threshold would be considered as initial guess for the total number of fast modes \n(e.g. 1e0)'))
nFastModeGuess = []
for iTime in range(nTime):
    nFast = 0
    for i in range(nSpecies):
        if Tau[iTime][i] <=TimeResolution:
            nFast+=1
    nFastModeGuess.append(nFast)
plt.plot(nFastModeGuess,'s')
plt.grid()

# End of Decompostion of phase space.........


# performing the refinement of intial trial basis vector (two-step recursive refinement procedure)

# import time
# start = time.time()
# a, b, dadt, dbdt = [], [], [], []
# for iTime in range(nTime):
#     # Computing the eigen values(w) and eigen vectors (v) of the Jacobian
#     w, v = np.linalg.eig(J_val[iTime])
#     a.append(v)
#     b.append(np.linalg.inv(v))
   
# for iTime in range(1, nTime):
#     dadt.append(a[iTime]-a[iTime-1])
#     dbdt.append(b[iTime]-b[iTime-1])
   
# a_refined, b_refined, tau_refined = [], [], []
# for iTime in range(nTime-1):
#     a_initial = a[iTime]
#     b_initial = np.linalg.inv(a_initial)
#     a_, b_, tau = refinement(a_initial, dadt[iTime], dbdt[iTime], nFastModeGuess[iTime],conc[iTime,:])
#     a_refined.append(a_)
#     b_refined.append(b_)
#     tau_refined.append(tau)

# end=time.time()
# print('Refinement finished in {} seconds.'.format(end-start))
# nTime = nTime-1


a_refined, b_refined = [], []
for iTime in range(nTime):
    # Computing the eigen values(w) and eigen vectors (v) of the Jacobian
    w, v = np.linalg.eig(J_val[iTime])
    a_refined.append(sym.re(sym.Matrix(v)))
    b_refined.append(sym.re(sym.Matrix(np.linalg.inv(v))))

#.................End of refinement of basis vectors.................................................................

# The criterion used here to determine the number M of exhausted modes is based on
# an error vector Yerror. Equation 6 from Valorani paper: M. valorani, Combustion and Flame 146 (2006) 29–51: 

f = np.zeros((len(conc),nSpecies),dtype=float)
for iTime in range(nTime):
    for i in range(nSpecies):
        dict_ = dict(zip(species_list,conc[iTime,:]))
        f[iTime, i] = np.array((b_refined[iTime][i,:]*g).subs(dict_),dtype=float)
epsilon_rel = float(input('Enter a value in the range [1e-1-1e-3] (epsilon relative) ;\n(e.g. 1e-2 (typical), )'))
_ = float(input('Enter a value in the range [1e-10-1e-14] (epsilon absolute) ;\n(e.g. 1e-14 (typical), )'))
epsilon_abs = _*np.ones((nSpecies,),dtype=float)

y_error = []
for iTime in range(nTime):
    dict_ = dict(zip(species_list,conc[iTime,:]))
    y_vec = np.array([species.subs(dict_) for species in species_list],dtype=float) 
    y_error.append(epsilon_rel*np.absolute(y_vec)+epsilon_abs)
    
# implementing equation 8 of Valorani paper M. valorani, Combustion and Flame 146 (2006) 29–51

nFastMode = []
for iTime in range(nTime):
#     print('Time stamp: {}'.format(iTime+1))
    for i in range(sum(~np.isinf(Tau[iTime]))-1,-1,-1):
        sum_ = 0
        for r in range(i):
            sum_+= (np.array(a_refined[iTime][:,r],dtype=float))*f[iTime,r]
        if(np.absolute(Tau[iTime][i+1]*sum_) < y_error[iTime]).all():
            break
        n = i+1
    nFastMode.append(n)
#     print('Number of fast modes: {}\n'.format(n))

plt.plot(nFastMode,'s')
plt.grid()


# End of evalaution of number of exhausted modes................................................................


# Evaluation of importance index from M. valorani, Combustion and Flame 146 (2006) 29–51

start = time.time()
I_fast, I_slow = [], []
for iTime in range(nTime):
    I_f, I_s = importanceIndex_Valorani(a_refined[iTime], b_refined[iTime], nFastModeGuess[iTime], conc[iTime,:])
    I_fast.append(I_f)
    I_slow.append(I_s)
del I_f, I_s
end = time.time()
print('Finished in {} seconds.'.format(end-start))


# End of evaluation of importance index.........................

# Start of Determination of projection matrix..........................................................
Q, radicalSpecies_list, nonRadicalSpecies_list, radIdx = [], [], [], []
for iTime in range(nTime):
    Q_, rad, nonRad, radIdx_ = projectionMatrix(a_refined[iTime], b_refined[iTime], nFastModeGuess[iTime], conc[iTime,:])
    Q.append(Q_)
    radicalSpecies_list.append(rad)
    nonRadicalSpecies_list.append(nonRad)
    radIdx.append(radIdx_)
    print('Radical species: {}\nNon radical species: {}\n'.format(radicalSpecies_list[iTime], nonRadicalSpecies_list[iTime]))


# End of Determination of projection matrix..........................................................


#.............................Start of Simplification algorithm based on valoriani 2006...................................	

# The simplifications achieved by the CSP-derived reduced model depends
# not only on the user-specified error tolerance thresholds, but also on the
# user’s selection of the set of primary species of interest

print('Simplification Algorithm.....................\n\n')
_ = list((-1)*np.arange(0,30,dtype=float)[::-1])
tolerances = [pow(10,i) for i in _]

S_global, R_global = [], []

for tol in tolerances:
    S, R = simplificationAlgorithm(nTime, kernel, tol)
    S_global.append(S)
    R_global.append(R)   
    
for i, tol in enumerate(tolerances):
    plt.semilogx(tol,len(S_global[i]),'sb')
plt.grid()
plt.xlabel('tolerance')
plt.ylabel('# active species')
plt.show()
for i, tol in enumerate(tolerances):
    plt.semilogx(tol,len(R_global[i]),'sg')
plt.grid()
plt.xlabel('tolerance')
plt.ylabel('# active reactions')
plt.show() 


#.............................End of Simplification algorithm based on valoriani 2006...................................	
		
		
#.............................Start of Simplification algorithm based on Lam and Goussis paper 1993...................................

# The simplifications achieved by the CSP-derived reduced model depends
# not only on the user-specified error tolerance thresholds, but also on the
# user’s selection of the set of primary species of interest


start = time.time()
P = []
for iTime in range(nTime):
#     print(iTime)
    _P_ = participationIndex_Goussis(b_refined[iTime], nFastModeGuess[iTime], conc[iTime,:])
    P.append(_P_)
del _P_

print('Summation of absolute values........\n')
for iTime in range(nTime):
    print('at time: {}'.format(iTime))
    for iSpecies in range(nFastModeGuess[iTime]):
        print('For species {}: {}'.format(species_names[iSpecies], np.sum(np.absolute(P[iTime][iSpecies,:]))))

I = []
for iTime in range(nTime):
#     print(iTime)
    _I_ = importanceIndex_Goussis(a_refined[iTime], b_refined[iTime], nFastModeGuess[iTime], conc[iTime,:])
    I.append(_I_)
del _I_

print('Summation of absolute values........\n')
for iTime in range(nTime):
    print('at time: {}'.format(iTime))
    for iSpecies in range(nFastModeGuess[iTime]):
        print('For species {}: {}'.format(species_names[iSpecies], np.sum(np.absolute(I[iTime][iSpecies,:]))))
end = time.time()


print('Simplification Algorithm.....................\n\n')
_ = list((-1)*np.arange(0,30,dtype=float)[::-1])
tolerances = [pow(10,i) for i in _]

S_global, R_global = [], []

for tol in tolerances:
    S, R = simplificationAlgorithmGoussis(nTime, kernel, tol)
    S_global.append(S)
    R_global.append(R)   
    
for i, tol in enumerate(tolerances):
    plt.semilogx(tol,len(S_global[i]),'sb')
plt.grid()
plt.xlabel('tolerance')
plt.ylabel('# active species')
plt.show()
for i, tol in enumerate(tolerances):
    plt.semilogx(tol,len(R_global[i]),'sg')
plt.grid()
plt.xlabel('tolerance')
plt.ylabel('# active reactions')
plt.show() 

#.............................End of Simplification algorithm based on Lam and Goussis paper 1993...................................



