# computational-singular-perturbation
computational singular perturbation 

# Computation singular peturbation method implemented in 
# Python programming langauge by Dr. Harish Kumar Chakravarty and Prof. Alexis Matynia, Prof. Patrick De Costa, and Prof. Anca Belme at 
# Jean Le Rond d'Alembert Institute - Sorbonne University www.dalembert.upmc.fr

# User must enter the name of input file here (such as param_general.ini)

# Refinement functions: The basis vectors generated from the Jacobian matrix provide 
# leading order approximation and has been used as intial trial basis vectors in the refinement procedure. 
# Since basis vectors changes as a function of time, it is important to perform two step
# CSP refinement procedure to generate the correct refined set of vectors valid at all the time (time-dependent basis vector).
# In general, CSP method provides programmable two-step recursive procedure for refinement of the linearly  independent basis vectors 
# obtained from the Jacobian matrix of chemical kinetic mechanism to generate the correct refined
# set of orthogonal time-dependent basis vector to develop a reduce model with higher accuracy
# Combustion Science and Technology, 89, 5-6, p. 375, 1993. 

