# ECSE-608-Project
Automatic bolus correction module for artificial pancreas control systems.

Augmenting the artificial pancreas with a safety agent which detect when a missed or underestimated mealtime bolus and then provides automatic bolus correction may improve the patient safety and the artificial pancreas efficacy.

The safety agent should continuously observe the patient state and action, understand the patient habits and model and then detect when the patient underestimates or misses the meal bolus. The space of all possible situation that the agent should understand is infinitely huge and the patient model is highly non-linear. This motivates the use of machine learning to train an agent (a neural net) to make good decisions.
 
The purpose of this work is to train an agent using reinforcement learning techniques in order to provide automatic bolus correction for artificial pancreas control systems.

Please find instructions in the main file is DeepNetBolusing.m.
Anas
