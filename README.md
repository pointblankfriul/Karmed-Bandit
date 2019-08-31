# Karmed-Bandit
K-Armed Bandit implementations

master:
  - KarmedStationary with or without UCBAS
  - KarmedNonStationary
  - Gradient Karmed for both Stationary and non-Stationary problems
  
  All subclasses have the following methods:
    - setTrainParams() to set iterations, epsilons
    - run() to execute the model and start the training
    - setRandomNoise() which is just for non-stationary problems
  
  Parameters needed for Instantiation and the functions above are explained 
  in each function description.
  
feature/0.ucb:
  - Implements the upper confidence bound action selection method for 
    the stationary problems
    
feature/1.gradient:
  - Implements the Gradient bandit algorithm for both stationary and non
    stationary problems
