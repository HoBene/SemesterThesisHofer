Github repository belonging to Semester Thesis written at the Chair of Thermofluiddynamics, Technical University of Munich 
Title: SINDy - a Promising Machine Learning Approach to Model Flame Dynamics
Author: Benedikt Hofer (bendikt.hofer@tum.de)
Supervisors: M.Sc. Axel Zimmermann and Prof. Wolfgang Polifke, Ph. D 


What each file does....

Summary.csv: Summary of all modeled ODEs' accuracies and corresponding savgol filter window size
 
Explicit_solver.mat: MATLAB Explicit ODE solver
Implicit_solver.mat: MATLAB Implicit ODE solver 

AE.py: Create, Train and Test standard Autoencoder
VAE.py: Create, Train and Test variational Autoencoder

get_signals_y.py: Divide flame into multiple signals along y axis 

1D_Sindy.py: SINDy for explicit and implicit ODEs modeling Doehner data. Relates to one dimensional modeling approach
sindy.py: SINDy for modeling Eder data. Relates to modeling one dimensional and one dimensional divided flame dynamics

SindyAE.py: SINDy for modeling latent space discovered by standard Autoencoder
SindyVAE.py: SINDy for modeling latent space discovered by variational Autoencoder


some additional comments are provided within each file  
