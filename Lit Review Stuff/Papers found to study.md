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
	- ***This paper is very good for providing an overview of methods people have taken. May want to use it as a basis for more research in the future*
	- Believes one of the main obstacles is the lack of BG prediction models that are reliable enouh to model the varience of a diabetic patient's physiology. We should be able to mimic the patient's physiology and cope with external factors like noise, exercise, stress, unannounced meals etc.
	- Believes we need an updated review to establish current trends in modeling strategies.
	- Thinks there is no common consensus about how to include physical activity and other intra-patient variablity sources in the glucose kinetics
	- **Predictive Models placed into 4 different categories**
		- **Physiological models**
			- Require a previous understanding of insulin and glucose metabolism.
			- Useful for performing simulations of BG metabolism in the form of compartmental models and for studying the physological processes that are involved in glucose regulation.
			- ![[Pasted image 20221101133716.png]]
			- Major drawback is that this type of model contains several physiological parameters that need to be set prior to their use to make BG predictions. They can be adjusted using identification techniques, ML techniques, or population values. 
			- Difficult to find a satisfying model with a good generalisation capability as they usually contain several variables and parameters that are difficult ot adjust.
		- **Data-driven models**
			- Fully rely on CGM data and, sometimes, additional signals to model a patient's physiological response without involving physiological variables.
			- E.g. using NN / Autoregression models.
		- **Hybrid Models**
			- A physiological model for glucose digestion and absorption, a second model for insulin absorption , and a third model for exercise. These models are typically used in pre-processing.
			- Usually known as hybrid models as they partially rely on physiological models and require the identificatoin and setting of some physiological parameters.
		- **Control-relevant models**
			- Used in internal-model control algorithms and can use any of the prior alternatives.
	- **Physiological Model Results**
		- Two types of physiological models 
			- **Minimal models** - capture crucial processes of glucose metabolism and insulin action with few equations and identifiable parameters.
			- **Maximal / Comprehensive models** - all the available knowledge of the physiological system and are capable of simulating or reproducing a diabetic patient's metabolic response, which allows experiments to assess controllers and treatments.
		- Most popular models are compartmental models forming a number of interconnected compartments to try and describe the processes that occur in the inaccessible parts of the system.
		- Most popular proposals for physiological models of insulin action and glucose kinetics are the **Dalla Man Model, Hovorka Model and Bergman minimal model**.
			- For these models, the input variables include factors from external insulin therapy and nutritional content over time.
			- For works in BG prediction, the breakdown of models used was as follows:
				- Hovorka - 33%
				- Bergman/modified - 25%
				- Dalla Man - 8.3%
				- Others - 33%
			- *Perhaps do some further research into these compartmental models when looking into prediction.*
	- **Data-driven Models**
		- Trends show that most researchers are experimenting with a vast pool of ML techniques.
		- Many models for forecasting glucose concentration use several inputs, some works suggest that ingested carbohydrate information, alpng with injected insulin might be redundant. This is why many approaches just use CGM data as an input.
		- Other studies state that the use of additional inputs makes the prediction task harder because formalising these inputs in mathematical terms and extracting useful signals from them is not easy.
		- Around 33% of data-driven models use mixed techniques.
	- **Hybrid Models**
		- Use both data-driven and physiological models. Usually a physiological model followed by a data-driven model that learns the relationship between inputs and future outcomes.
		- Physiological is frequently meal models and insulin absorption models. 
		- Most popular meal absorption model currently is the Dalla Man. NN is the most common data-driven prediction model.
	- PH range of 15-120 min is usually explored, and a 30 min PH is the most common value.
	- Other signals being added to prediction approaches are heart rate, perceived exertion rate and sleep. Still believes that there are more signals that affect BG levels like emotional state and some illnesses.
	- Most popular performance metrics are always defined in terms of the error. e.g. Mean squared error, sum of squared errors.
		- This doesn't treat errors differently for hypos/hypers so some mestrics such as glucose-specific MSE have been proposed to add extra penalties whenever the error is potentially more dangerous from a clinical point of view. Quite like Clark's error grid.
	- Relative absolute different and coefficient of determination are often calculated and reported as standard metrics.
	- Many proposals lack clinical evidence as they are only validated with *in silico* data.
	- Clear trend for model individualisation because it allows adaptation of the model features and their relevancy on the prediction in terms of the particular physiology and lifestyle of the patient and obtaining predictions that are more accurate.

- Artificial Neural Network Algorithm for Online Glucose Prediction from Continuous Glucose Monitoring
	- Uses ANN. Inputs preceding 20 mins, output prediction of glucose concentration at the chosen prediction horizon time. Performance assessed over 3 PHs: 15, 30 and 45 mins. Accuracy estimated using the RMSE and prediction delay.
	- RMSE ~10, 18, 27 mg/dL for 15, 30 45 min respectively. Prediction delay is around 4, 9 and 14 min for upward tends and 5, 15 and 26 min for donward trends.
	- 10$^{th}$ order data-driven autoregressive model has been tested by Reifman and others and had quite accuracte predictions for a PH of 30 min. (*Could definitely be worth exploring AR models*).
	- Palerm et al. used a method based on the estimation of glucose and its range of change, using a **Kalman filter**.
	- Model is **only trained on CGM data** in this example.
	- **It was found convenient to reduce noise in the data by prefiltering them using a casual Kalman filtering method**. However, the assessment of the prediction in the evaluation section is done by taking the original data profiles as reference.
	- **NN architecture**
		- 3 layers; 10 neurons, 5 neuron, 1 neuron. 
		- Transfer function is sigmoidal in both layers. Neurons are totally connected and feed forward
. The output layer has a linear transfer function.
		- The NN takes in glucose measurements **up to 20 minutes before the current time**. As the sampling rate varies from one CGM system to another, the number of NN inputs is different for each dataset.
		- The ouptut of the network is the glucose prediction at the PH time.
	- **ARM architecture**
		- First order model with time-domain equation $u_i = au_{i-1} + w_i$ 
		- i = 1,2, ... , n denotes the order of glucose samples collectup up to the $n^{th}$ sampling time $t_n$, and $w_i$ is a random white noise process with zero mean and variance equal to $\sigma^2$.
		- Let $\theta = (a, \sigma^2)$ A new value of $\theta$  is determined by fitting past glucose data $u_n, u_{n-1}, u_{n-2}, ...$ by a weighted linear recursive least squares algorithm. Once $\theta$ is determined, the model is used to calculate the prediction of glucose level $Q$ steps ahead, where $Q.T_s = PH$ ($T_s$ is the the sensor sampling period).
		- All the past data participate with different relative weights, in the determination of $\theta$.
		- They chose exponential weights $\mu^k$ is the wieght of the sample taken $k$ instanst before the actual sampling time with $\mu$ termed the *forgetting factor*.
		- A different optimal $\mu$ value was chosen for each dataset and for each PH, minimising the RMSE between the original data of the training set and the predicted values obtained applying the ARM to it.
	- **Evaluation**
		- Model accuracy evaluated as RMSE of the predicted profiles vs originals.
		- Model delay estimated by calculating the delays between the original and predicted profiles when they cross three different thresholds defined in the following way. First peaks and nadirs are identified in the original profile. Then, threhsolds are placed at 25%, 50%, and 75% of the nadir-to-peak / peak-to-nadir distance for positive/negative trends.
		- The final model delay is calculated as the average of threshold delays for every positive and negative trend.
	- **Results**
		- ARM seems to before better at shorter PHs and has less delay overall.
		- NN performs better with longer PHs.
		- **Both models are more rapid in upward than in downward slopes.**
	
- A Real Time Simulation Model of Glucose-Insulin Metabolism for Type 1 Diabetes Patients
	- Based on compartmental models -> recurrent neural network
	- Compartmental models produce estimations about 
		- The effect of a short acting insulin intake
		- The effect of intermediate acting insulin
		- The effect of carbohydrate intake on blood glucose absorption from the gut.
	- Argues most physiological models as data sources can only be used for educational purposes as they ignore a number of factors associated with glucose metabolism.
	- CMS passed as inputs to NN along with current BGL.
	- **Mathematical models**
		- After the injection of D units of insulin, the change in plasma insulin concentration I is given as:
			- ![[Pasted image 20221104181505.png]]
			- where $k_e = 5.4 lt/h$ is the first-order rat eof constant insulin elimination, $V_i = 9.94 lt$ is the volume of insulin distribution, and $T_{50}$ is the half-time insulin dose absorption.
			- **VIEW THIS LINK FOR LOOKS INTO MATHEMATICS WHEN BUILDING CMS**
			- https://ieeexplore.ieee.org/abstract/document/1616403
	- RNN is fully connected, and contains update on-line the RNN weights. **RESEARCH INTO THIS**. Two strategies applied for comparative reasons - Free-Run (FR), and the Teacher-Forcing (TF).
	- 3 layer NN with tan-sigmoid and linear activation functions. Inputs for the available data, inputs of the NN, have been normalised for unity standard deviation and zero mean.
	- RMSE and Correlation Coefficient to measure accuracy. 
	- Results from the RTRL-FR method are superior to those obtained by the RTRL-TF method.
	- **Accuracy for this method was very promising**.
	
- Blood glucose prediction model for type 1 diabetes based on artificial neural network with time-domain features
	- ANN, time-domain attributes to predict blood glucose levels 15, 30, 45, 60 min in the future.
	- Features are previous 30 min of BG measurements before a trained model is generated for each patient.
	- Compares against many other data-driven BG predictors and claims it outperforms.
	- Believes that combining time-domain attributes into the input dataresulted in enhanced performance of most prediction models.
	- *Might be worth looking at Gaussian processes / eXtreme Gradient Boosting*.
	- **CGM values are the only input**. To improve prediction accuracy, time-domain features are also used.
	- **Data preparation**
		- Time-series dataset conerted into a set of paired inputs and desired outputs.
		- Uses the 'direct method' to perform prediction of each horizon independently from other prediction. Returns a multi-step forecast by concatenating all of the predictions made.
		- To perform this, a **sliding window approach (LOOK INTO THIS)** used for segementation of the time series dataset.
	- **Time-domain features and prediction model**
		- Use minimum, maximum, mean, std, peak to peak amplitude, median, kurtosis and skewness as time-domain features. Extracted from each window S generated by the sliding window approach.
		- These time-domain features are appended to the matrix of data X to create a matrix of all the glucose readings within the window as well as the time-domain features.
		- Uses a Multi-Layer Perceptron (MLP) model to predict blood glucose. Fully connected, multiple hidden layers, one node output layer. Each unit has its own bias.
		- Back-propagation to calculate gradients. Mean squared error between prediction and target values.
		- **Grid search algorithm (LOOK INTO)** to automatically select the best parameters for the proposed MLP model.
	- **Data cleaning**
		- All missing data was imputed through spline interpolation.
		- Savitzky - Golay technique to filter noise.
		- 80% train/test split.
		- Features normalised using min-max scaling.
		- **Uses a glucose-specific metric introduced by Favero et al.** called **gMSE** that applies specific pentalties to MSE.
		- This penalty function penalises an overestimation in hypoglycemia, and an underestimation in hyperglycemia.
	- Models in general performed better for shorter range PH.
	- Found that customising window sizes etc. for each patient improved the prediction performance.
	- A **recursive strategy** can be made for predictions where you recursively make shorter predictions to generate a longer term prediction.

- Deep Physiological Model for Blood Glucose Prediction in T1DM Patients
	- Carbohydrate and insulin arsorption in physiological models are modeled using a RNN inplmeneted using LSTM cells.
	- Trains and validates on both simulated data using the AIDA diabetes software simulation program and with real patient data from the D1NAMO open dataset.
	- Hayeri added heart rate, step-count and insulin information to the BG signal. Proposed algorithm applied to 9 children and the model was able to the predict the user's future glucose values with a 93% accuracy for 60 min ahead of time.
	- Plans to train an RNN to figure out the impact of fast insulin, slow insulin and carb intake on BG levels. Combine this with RNN using CGM data to try and obtain an accurate prediction on the variation in BGL.
	- ![[Pasted image 20221107151426.png]]
	- Question marks represent the number of time samples fed into the model. A time span of 9 h is being used and the simulated data will produce data samples every 15 min. Therefore 36 samples per 9h window.
	- As future values for meals and insulin aren't fed into the model, the model will generate an estimate about what will happen to the glucose signal if no external action is taken by the patient. Therefore, the model could be used to warn the user in advance about negative episodes if no action is taken and recommend particular actions to avoid such episodes.
	- **Clarke Error Grid** used for analysis (**Seems Important**).
	- *Follows the general trend where models appear to be able to predict rising BG levels better than falling levels*.
	- **Model performs noticeably worse on real patient data than the simulated data**.
	- https://www.mdpi.com/1424-8220/20/14/3896/htm; **There is an appendix with model code**
	
- A Deep Learning Algorithm For Personalized Blood Glucose Prediction
	- Uses a **CNN** !! And Ohio dataset
	- RMSE to measure error.
	- Converts the task into a classification task where the change between the current and future glucose value is split into 256 different categories. Prediction results over a 30 minute PH.
	- Doesn't use many of the fields in the Ohio dataset as they were found to increase variance and reduce performance.
	- Replaces missing values with first-order interpolation. For testing data, first-order extrapolation is taken to ensure the future values are not involved. **The predictions of extrapolated intervals are ignored to guarantee that the result has the same length as the CGM testing data when evaluating the performance.**
	- *Introduces a part of the data with the longest continuous interval from other subjects and combines them into the current subject to form an extended training data.* This strategy keeps 50% of the data as the current subject and the other 5 patients contribute the other half.
	- Uses a median filter to remove noise at the cost of slightly raising the bias of the model. *This median filter is not used on the testing data*.
	- Believes the WaveNet model their CNN is based on is more time efficient for training and testing with smaller weights compared with RNNs.
	- **Model Components**
		- Main components in WaveNet are **causal convolutional layers.**
		- Uses **Dilated Convolutional Neural Netowrk layers** (**Maybe research but doesn't seem hugely relevant**) which increases the receptive field of the input signal. 
		- 3 DCNN blocks with each containing 5 layers of diluted convolution.
		- ReLu activation function where $ReLu(x) = max(x, 0)$.
		- **Sparse softmax cross entropy cost function for optimisation.**
		- Uses **adaptive moment estimatoin optimiser** to adjust training steps.
	- **Results**
		- RMSE (AGAIN) for error calculation (*Lots of researchers use RMSE but could be interesting to use time within range/clarke error grid as well)*.
		- Curve fluctuates a lot around insulin/meal events (struggling to properly learn the impact of these events).
		- Lots of errors aroud the extrapolated regions of the data. First-order interpolation seemed to perform best for replacing regions of missing data. .
		- Subjects 575 and 591, predictions perform much worse. Large gap in training dataset. The data of these patients also fluctuates a lot more.
		- Differences between patients that use "humalog" vs "novalog" insulin.
		- Overall the model seems to have worse accuracy than deep learning models studied earlier, but is much more efficient.

- Blood Glucose Prediction With VMD and LSTM Optimized by Improved Particle Swarm Optimization
	- **This paper is way beyond my pay grade**.
	- Uses **Variational Model Decomposition??** to decompose and obtain the **intrinsic mode functions ??** of blood glucose components in different frequency bands, so as to reduce the non-stationarity of blood glucose time series.
	- Model seemed to perform better than LSTM, VMD-LSTM and VMD-PSO-LSTM methods.
	- **Variational Model Decomposition????**
		- **Maths too hard**
		- Finding the optimal solution of the variational problem.
		- Involves 3 concepts : classic Wiener filtering, Hibert transform and frequency mixing.
	- **LSTM**
		- LSTM is an RNN which learns long-term dependent information and avoids the problem of gradient disappearance.
		- In the hidden layer neurons of the RNN, LSTM adds a structure called a memory cell to remember past information, as well as three types of gate (input / forget / output) to control the use of historical information.
		- Key part is cell state $C$ which keeps the cell state storage at time t, and the cell state storage is adjusted by the forget gate $f_t$ and input gate $i_t$. The forget gate is for the celll to remember or forget its previous state $C_{t-1}$; the input gate will allow or prevent the input signal from updating the unit state; the output gate is the output of the unit state C and is transmitted to the next cell.
		- To allow the LSTM to predict a linear regression layer is added:
		- $y_t = W_{y0}h_t + b_y$, where $y_t$ represents the output of the final prediction, $b_t$ is the threshold of the linear regression layer.
	- **Particle Swarm Optimisation**
		- Initialises to obtain a set of random solutions and then iterates and finds the optimal solution by tracking the best particles in the current space.
		- Particles update their positions based on the best value they can get and the best value of their entire group.
		- Improved by adding a nonlinear variable inertia weight. Helps the model to converge faster on the optimal point.
		- This particle swarm optimisation is used to determine the optimal hyper parameters of the LSTM in order to match the network structure with the characteristics of the glucose concentration data.
	- **Steps of model training**
		1. Init parameters e.g. population size, number of iterations, learning factors and the limited interval of location and speed.
		2. Initialise the position and velocity of the particles.
		3. Determine the evaluation function of particles. Assign ecah particle to one of the parameters of the LSTM. Trains with training set, then verifies with the verfication set; taking into account both the training sample error and the verification sample error.
		4. Calculate fitness value for each particle position. The personal best and the group best are determined according to the initial particle fitness, and the best position for each particle is taken as its historical best position.
		5. During each iteration, update the particle's own speed and position through personal and global best.
		6. After satisfying the max number of iterations, input the prediction data and output the prediction value.
	- https://public.jaeb.org/direcnet/stdy/; Source of data, open for download.
	- Uses **RMSE and Clarke Error Grid** to evaluate performance.
	- The **IPSO** step shows significant improvements for the 45/60 min PH versus the normal **VMD-LSTM** memory; believes the IPSO gives better otimisation and better hyperparameters for the LSTM.
	- IPSO performs better than PSO optimisation (to be expected due to the faster convergence due to the variable inertia weight).
	- **Conclusions**
		- The IPSO leads to a largely increased computational load
		- Need to do more work on incorporating more patient-specific characteristics for better long-term predictions.

- A Multi-Patient Data-Driven Approach to Blood Glucose Prediction
	- Interested in finding methods that have been learned from multiple patients and are applicable to completely new patients.
	- Tried the approaches on a Non-Linear Autoregressive (NAR) neural network and on LSTM networks.
	- Compares to three literature approaches, based on feed-forward neural networks, autoregressive models, and recurrent neural networks.
	- NAR obtained good ccuracy for short-term predictions only.
	- LSTM did very well for both short- and long-term glucose-level inference.
	- Believes they can improve the performance by considerably enlarging the dataset used for learning the model (by combining all patient data).
	- **Pre-Processing**
		- Applied **Tikhonov regularisation** (**RESEARCH into this as we may want to apply)**, which is widely used in time series analysis and in glucose level prediction.
	- **Non-Linear Autoregressive Neural Network**
		- NAR computes the value of a signal $y$ at time $t$ using n past values of $y$ as reressors, as follows:
			- $y(t) = f(y(t-1), y(t-2), ... , y(t-n)) + e(t)$
			- where $f$ is an unknown non-linear function and $e(t)$ is the model approximation error at time $t$.
		- Function f(.) is computed using a multi-layer NN. At time $t$ the NN is fed with the $n$ past values of the signal $y$. 
		- **Levenberg-Marquardt** backpropagation procedure (LMBP) used for backpropagation.
		- As it doesn't need to calculate Hessian matrices, it helps to reduce training times.
		- Uses **optimal Brain Surgeon method** to prune the initial fully-connected structure of the network and (hopefully) reduce computational overheads.
		- Uses **Lipschitz quotients (Research if considering autoregressive networks)** to determine the optimal number of regressors for the model.
	- **LSTM Architecture**
		- Avoids vanishing/exploding gradient problem
		- Layer of 30 LSTM units and a single ouput layer (dense), with a number of units equal to the future glucose samples that need to be predicted).
		- To optimise hyper parameters and prevent under/over-fitting used an initial learning rate of 0.001, with dropout. At each stage, individual nodes and corresponding links are randomly dropped out of the model, leaving a reduced network
		- Helps to reduce co-dependency of network parameters and hence overfitting.
		- **Adaptive Moment Estimation (Adam)** for optimisation. Leverages adaptive learning rates to set individual learning rate per each parameter. We can then combine this with techniques such as stochastic gradient descent; proved to be particularly suitable for non-stationary time-series with a lot of noise.
	- Test set **completely independent from training set.**
	- **Analytical Assessment**
		- **Time lag** - prediction delay, defined as minimum time-shift between the predicted and observed signals which provides the heighest correlation coefficient between them.
		- **FIT** - the ratio of RMSE and the root mean square difference between the observed signal and its mean value.
		- **All models benefitted from the pre-processing of the training data with Tikhonov.**
		- **The LSTM obtained by far the best prediction performance**
			- Normal Feedforward NN was the worst.
			- NAR, AR and RNN methods were comparably successful.
			- RMSE gradually increased with regards to prediction time.
			- Prediction lag didn't occur in the LSTM until a 60 minute PH was introduced.
	- **Clinical Assessment**
		- Uses **Clarke Error Grid** to perform a more clinical assessment of the models. (A and B **clinically acceptable**)
			- **A**: values within 20% of the reference.
			- **B**: values that, in spite of being outside 20% of the reference, do not lead to inappropriate treatment of the patient.
			- **C**: values leading to inappropriate treatment, but without dangerous consequences for the patient.
			- **D**: values leading to potentially dangerous failure to detect hypoglycaemic or hyperglycaemic events.
			- **E**: values leading to treat hypoglycaemia instead of hyperglycaemia and vice-versa.
		- ~95% within zone A for 30 min PH.
		
- Application of improved LightGBM model in blood glucose prediction
	- Improved **LightGBM** model based on **Bayesian hyper parameter optimisation**.
	- RMSE of 0.5961
	- **LightGBM Model**
		- LightGBM is mainly a decision tree based on gradient-based one-side sampling (GOSS), exclusive feature bundling (EFB) and a histogram and leaf-wise growth strategy with a depth limit.
		- **GOSS** - keep all of the large gradient samples and perform random sampling on the small gradient samples accoridng to a proportion.
		- **EFB** - divide the features into a smaller number of mutually exclusive bundles, that is, it is impossible to find an accurate solution in polynomial time. Approximates the solution by selecting a small number of sample points that are not mutually exclusive and allowed between features.
		- **Histogram algorithm** - discretise continuous floating point features into $k$ integers, and construct a histogram with width $k$ at the same time.
		- LightGBM uses a leaf-wise growth strategy with a depth limit to find a leaf node with the largest split gain in all of the current leaf nodes, then splits etc.
		- **Some complicated maths, if we decide to do implement this it'll be worth going through some proper tutorials.**
	- **Bayesian Hyper-Parameter Optimisation Algorithm**
		- Establish a substitute function based on the evaluation result of the past objective to find the minimum value of the objective function.
		- The substitute function established in this process is easier to optimise than the original objective function, and the input value to be evaluated is selected by applying a certain standard to the proxy function.
		- Should reduce the amount of time it takes the model to converge by taking into account the previous evaluation when trying a new set of hyperparameters.
		- Obtained optimal parameters with MSE as the evaluation indicator.
	- **Improved LightGBM Model Based on Bayesian Hyper-Parameter Optimisation Algorithm**
		1. Divide the dataset into training/testing, process missing values, analyse the weight fo the influence of the eigenvalues on the results, delete useless eigenvalues, delete outliers.
		2. Use the Bayesian hyper-parameter optimisation algorithm for the parameter optimisation of the LightGBM model, and the HY_LightGBM model is constructed and trained.
		3. Use the HY_LightGBM model for prediction and output the results.
	- Analysed missing values in the original dataset to obtain the proportion of the missing data. Deleted features with large amounts of missing values.
	- Uses eigenvalue analysis **(?)** to remove invalid eigenvalues and improve accuracy of the experiment by ignoring unrelevant features.
	- Evaluated prediction with MSE, RMSE, and determination coefficient R2 (R-Square). 
	- Seemed to perform very well for prediction (**RMSE 0.7721**)
	- The hyperparameter optimisation significantly increases the time for the model to be trained, but does also noticeably improve prediction accuracy.
	- This model also outperformed XGBoost and CatBoost.
	- Other parameter optimisation algorithms include **Genetic Algorithms** and **Random Searching Algorithms**.
	- Believes this model has a much stronger generalisation ability than others, and doesn't need to be retrained.

- Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks

Start Writing up.
Write out what you searched, what you found out and the conclusions you drew from that.
![[Pasted image 20221107113116.png]]
![[Pasted image 20221107133425.png]]

## Gradient Boosting

Term 'gradient boosting' comes from the idea of 'boosting' or improving a single weak model by combining it with a number of other weak models, in order to generate a collectively strong model.
**Gradient Boosting** is an extension of boosting where the process of additively generating weak models is formalised as a **gradient descent** algorithm over an objective function.
Is a **supervised learning algorithm**.
Often uses **decision** trees as the model for the gradient boosting.

**Residual** - value obtained when subtracting the predicted label from the true label.
Each iteration, we try and predict the residuals by adding new trees to the main tree. By doing this, we are effectively performing a gradient descent algorithm on the squared error loss function for the given training instances.

### XGBoost
XGBoost also uses second-order gradients of the loss function in addition to the first-order gradients, based on a Taylor expansion of the loss function.

Ends up forming a more sophisticated objective function containing regularisation terms. This extension of the loss function adds penalty terms for adding new decision tree leaves to the model with penalty proportional to the size of the leaf weights. These penalty terms inhibit the growth of the model to prevent overfitting.