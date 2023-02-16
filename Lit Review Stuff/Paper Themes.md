1. RNN
	- RMSE
	- AIDA maths model
	- 2 layer RNN.
2. Overview, look deeper for examples
3. ANN + ARM
	- CGM data
	- causal Kalman filter
4. Compartmental -> RNN
	- Has update on-line weights
	- 3 layers
5. ANN with time domain features. CGM values input.
6. Deep learning to create physiological models. Combined with LSTM RNN
7. CNN - changes it into a classification task where the change between current and future glucose is split into 256 different categories.
8. VMD - LSTM - optimised with particle swarm optimisation.
9. Autoregressive NN and LSTM network. Trains on multiple patient data, thus enlarging the dataset.
10. Decision tree based on gradient-based one-side sampling.

We can tell a story; basic ANN -> ANN with time features -> RNN -> Compartmental + RNN -> Deep Physiological + RNN -> VMD LSTM -> Multiple Data -> CNN -> Decision Tree