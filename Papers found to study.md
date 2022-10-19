- Blood glucose prediction using artificial neural networks trained with the AIDA diabetes simulator: a proof-of-concept pilot study --> *February 2011*
	- Elman Recurrent ANNs used to make BGL predictions based on a history of BGLs, meal intake and insulin injections.
	- Nocturnal periods were found to be more accurate.
	- Accuracy measured as the **root mean square error (RMSE)** over 5 test days.
	- St deviation over a 10 hour period was 4 times larger than over a 1 hour period.
	- Control algorithms tend to be grouped into 2 broad categories:
		- Proportional-integrial derivative (PID) controller.
			- Classic feedback approach comparing the difference between the measured glucose level and the expected glucose level. (Known as the **proportional component**)
			- **Integral component** - area under the curve between the measured glucose level and the expected glucose level.
			- **Derivative component** - rate of change of the measured glucose level.
			- **Insulin Infusion Rate (IIR)** - rate at which insulin is supplied to the body.
			- We can caluclate IIR with the equation below where $K_P, K_I, K_D$ are the proportional, integral and derivative weight terms, respectively, and $G, G_t$ are the measured and expected (target) glucose levels, respectively:$$IIR = K_P(G - G_t) + K_I\int(G-G_t)dt + K_D.\frac{dG}{dt}$$
		- Model predictive control (MPC).
			- Effects of carbohydrate intake and insulin injections on the BGL are captured in aglucoregulatory model.
			- Can be as a mathematical description of the various physiological processes of the body which can affect the BGLs.
			- Could be formulated as a 'black box' approach, relying on pattern recognition from sets of training data.
			- *Currently uses this 'black box' approach*
	- Claims intersubject variability is one of the major challenges to be addressed in any closed loop system. Argues that ANNs may solve this as they can be trained on an individuals BGL.
	- Recurrent ANNs often chosen for superior performance in time-series prediction problems.
	- **Previous Approaches**
		- Sandham tried to use ANNs - hindered largely by a lack of data.
		- Mougiakakou devleoped a sim model based on 3 **compartmental models (CMs)** describing short- and long-acting insulin effects as well as a model for glucose absorption from the gut. These models were then fed as inputs into an ANN along with 
		- This same approach was tried again using a real-time RNN and a feed-word network. Networks developed specifically for each patient. Both had similar performance, but RNN preferred as it could adapt weights when a new input was supplied.
		- Zarkogianni used an ANN and CM; made short term predictions which were then fed into a nonlinear predictive controller capable of advising on insulin doses. System was able to control BGLs with realistic meal intakes.
		- Pappada measured results as the mean absolute difference (MAD%) between the ANN prediction and the output from the CGM.
			- ![[Pasted image 20221018120300.png]]
			- Results showed predictions were more accurate for BGLs in normoglycaemic and hyperglycaemic ranges than those in the hypoglycaemic range.
			- **An increase in predictive length resulted in a decrease in predictive accuracy**.
		- Perez-Gandia used a feed-forward, fully-connected ANN with 3 layers. Inputs current BGL and the BGL readings up to 20 minutes prior to the reading. ONly using past CGM data as the input to the ANN limited the eprformance of the prediction system. Sugested that *including additional input data, such as meal intake and insuline dosage could improve the performance.*
		- Curran used Levenberg-Marquardt algorithm to make BGL predictions based on expected insulin doses, carb intake and exercise. Generates a database of previous BGLs, insulin doses, carb intakes and exercise which would be sent via a 3G connection to an endocrinologist to review the control of BGL of a given patient.
	- **Proposed iterative development of artificial pancreas**
		1. System to terminate insulin infusion from pump at low BGL to prevent hypo. Closed-loop system for nocturnal control.
		2. Glucose control around meal times with some user-interaction.
		3. Closed-loop control around meals and periods of exercise.
		4. Fully implantable systems or systems with dual control able to provide glucagon or other substances to prevent hypoglycaemic events.
	- Data from **AIDA**; a mathematical model and freeware diabetes simulator.
	- **Their implementation**
		- Used a 2 layer RNN with a feedback layer from the context layer to the input layer.
		- Tansig activation function in hidden layer, linear activaiton for output neuron.
		- Trained with the Levenberg-Marquardt algorithm. Training was assessed by MSE.
	- ![[Pasted image 20221019142919.png]]
	- **Their Discussion**
		- RNNs more accurate during the night due to impacts of meals and exercise etc.
		- Networks with few hidden neurons performed well at both short and long nocturnal predictions.
		- Some real life factors e.g. restless sleep, sexual relations, variations in meal content weren't included by the mathematical model used to produce data.
		- AIDA model also removes the effects of stress, exercise, illness, pregnancy etc. so simplifies the problem of BG prediction.
		- Believes that an ANN trained on more than one patient's data would be inappropriate and reduce accuracy. *(Perhaps worth testing a general model vs individualised models?)*
		- For short term, the NNs made predictions in short term with $RMSE_{5 \ day}$ of $0.15 \pm 0.04 \ SD\ mmol/L$ and an $error_{max}$ of $0.27 \ mmol/L$.
		- For long term (8 and 10 hrs) NNs made predictions with $RMSE_{5 \ day}$ of $0.14 \pm 0.16 \ SD \ mmol/L$ and had an $error_{max}$ of $0.20 \ mmol/L$ for 8 hrs, $0.36 \ mmol/L$ for 10 hours.
		- Future investigation; using real time recurrent networks which are capable of continuous learning. Could be an interesting thing to study for changing metabolism over a period of weeks and months.
- A review of personalized blood glucose prediction strategies for T1DM patients
	- 

- Artificial Neural Network Algorithm for Online Glucose Prediction from Continuous Glucose Monitoring

- A Real Time Simulation Model of Glucose-Insulin Metabolism for Type 1 Diabetes Patients

- Blood glucose prediction model for type 1 diabetes based on artificial neural network with time-domain features

- Deep Physiological Model for Blood Glucose Prediction in T1DM Patients

- A Deep Learning Algorithm For Personalized Blood Glucose Prediction

- Blood Glucose Prediction With VMD and LSTM Optimized by Improved Particle Swarm Optimization

- A Multi-Patient Data-Driven Approach to Blood Glucose Prediction

- Application of improved LightGBM model in blood glucose prediction

- Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks