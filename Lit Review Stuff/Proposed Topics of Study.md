**Preprocessing**
1. **Kalman Filtering?? to reduce noise in data**.
2. **Tikhonov Regularisation??** widely used regularisation method in time series analysis and in glucose level prediction. (*seems to be the better option of the two*)
3. **Spline interpolation** for missing data?
4. **First-order interpolation** for missing data? (*seems to be the better option of the two.*)

**Parameter Optimisation**
1. **Bayesian Hyper-parameter optimisation??**
2. **Adaptive Moment Estimation??**
3. **Grid Search Algorithm**

*When we come to building models have a look into these three, have a look at them and pick the one that seems the most relevant option.*

**Learning Techniques**
1. Start off with a simple LSTM (not RNN to avoid gradient explosion/disappearance) for prediction solely via CGM. Contrast this RNN with a basic **autoregressive mode**l to argue why we'll use NNs for the future. 
	1.  Vary different horizons of previous blood glucose levels considered. **20/40/60 min**?? How helpful is previous data? How does it tank performance?
	2. Interesting to try **on-line updating RNN**. Train both models on group of patients, then give it a new patient. Does the udpate on-line model manage to actually learn a new patient on the fly??
2. Incorporate 3 compartmental models; one for glucose, one for insulin absorption (fast acting), another for insulin (intermediate acting) and feed these into the LSTM. Does this improve performance??
	1. https://ieeexplore.ieee.org/abstract/document/1616403 <-- Maths for compartmental models is in this paper.
	2. **Dalla Man Model, Hovorka Model and Bergman minimal model; look into for compartmental models.**
	3. Might be worth considering **Deep Physiological Models if we need more to do.**
3. Incorporate time-domain features into the prediction model. Are features like this more useful in improving accuracy than the compartmental models?
	1. https://www.sciencedirect.com/science/article/abs/pii/S0208521620301248 <-- Maths for time-domain features is in here.
4. Incorporate everything into one model. Have we overcomplicated things now? Are we overfitting on the training data?

**Prediction Techniques (for longer PH)**
1. Predicting by just guessing the next PH.
2. Recursive prediction across 30 minutes.
3. Recursive prediction across 5 minutes.

*Does recursive prediction help accuracy? Is there a tradeoff point where too much recursion results in a reduction in accuracy?*

**Evaluation Techniques**
1. RMSE very common.
2. Clarke's Error grid for more specifics.
3. Look at model delay for the different predictions. Do certain models delay less than others?
4. How do models perform in shorter vs longer PHs? Do some recursively predict better?
5. How do they do at predicting Hypos vs Hypers? Certain models better than others?

**If we want to use another dataset, we have this link**:
https://public.jaeb.org/direcnet/stdy/