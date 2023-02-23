*Aiming for few thousand words??*
### Inclusions
Higher level overview of what Type1 Diabetes is. What a type 1 diabetic has to do on a day-to-day basis --> Leads into regulating is not an exact science.

Go into more detail about 'life threatening situations.' Hypo issue day-to-day vs hyper. Impacts of hyper vs hypo. Long term high blood glucose leading to issues like kidney damage, nerve damage etc. Sets context better on why regulating is really important.

Go into detail about where blood glucose needs to be. WHO reference on where BG needs to be. (**between 70 mg/dL (3.9 mmol/L) and 100 mg/dL (5.6 mmol/L)**) for non diabetics. 5-9 for diabetics.

This is condition, what people do, these are the issues -> focus of work and how it is going to help them. Talk briefly about the advent of tech in the industry ---> CGMS, pumps etc. Talk about how Blood glucose meters/ injections -> Pumps and CGM devices through the NHS. Throw in a few pictures. 

Break it down into headings

Use as a means of giving longer terms PH. If you could apply to CGM to give advance warnings to people.
Try and find a reference about PH and CGM, talking a bit about that. Gives time for food to take action -> makes the reactiveness of treating the condition easily. End up yo-yoing because of how poor the PH is; a better prediction horizon would enable much greater control.

- Condition, Talk a bit about what controls blood glucose; predom carbs. There are then other factors....
- What it does
- What's happend so far
- PH + use of ML why it might help
- Difficulty of this, often use BG, carbs, sometimes exercise; mood 48 factors that influence? Lots of them you never have access to. When is something good enough?? Might be worth talking a bit more in depth about feature extraction from these 48.
- Talking about some of the approaches people have looked at.

Intro which talks about the condition in a general sense. Follow it wiht a chapter which is your research. Specifically talk about machine learning and its use for the condition. As a part of the second chapter, talk about work that's been done reference papers. Refer to them in the discussion. 

At first step back from research; talk about why machine learning is important; what they did in the early days.

Finish intro with a little closing bit slight overview. Second bit need a start with just a little general sense talking about machine learning (not related to diabetes). 

## Introduction
*Supposed to contain information about Diabetes/the background of why the research is necessary??*

Type 1 Diabetes is a chronic disease that impacts people's ability to produce insulin - either partially or completely. As such, Type 1 diabetics are reliant on taking external insulin in order to regulate their blood glucose levels.
Unfortunately, regulating the body's blood glucose levels is not an exact science as there are many variables that can impact a patient day to day: be it their mood, what they have eaten or any exercise they may have done. This leads to many mistakes being made in the process, which can cause potentially life-threatening situations. With an estimated 422 million people (*according to the WHO*) suffering from diabetes worldwide, it is clear that there is a need for technology to streamline and automate the regulation of blood glucose levels, saving patients' lives in the process.

The increasingly widespread introduction of Continuous Glucose Monitoring (CGM) devices has lead to the creation of new time-series datasets which provide a deeper view into how a patient's blood glucose level fluctuates. There has been much work done applying Machine Learning (ML) techniques to these datasets, with the goal of being able to accurately predict future blood glucose levels and produce a closed loop Artificial Pancreas (AP) system that could regulate a patient's blood glucose levels without any outside input. However, whilst there has been some success in this endeavour, most successes have been seen in *in silico* trials, where the data is produced by a diabetes software simulation program. These simulation programs fail to capture the true complexity and variability of real patient data. In reality, when using real patient data, there are still many limitations to the prediction capabilities of these models and they are seemingly unable to consider the multitude of factors which can impact a patient's blood glucose levels, and handle inaccuracies in CGM readings. 

The three most pertinent ML techniques which have been applied to the task of blood glucose prediction are Autoregressive Models (ARM), Artificial Neural Networks (ANN) and Gradient Boosting Models (GBM). All of these techniques have seen good results in short-term blood glucose prediction, and in this review we will be discussing the different approaches taken to applying these different techniques to blood glucose prediction - comparing and contrasting their relative benefits and drawbacks.

## Main Body
*Go into approaches taken and what we have found out from the failures/successes of those different approaches??*


## Conclusion
*What can we draw from the approaches that we've seen?? How does it link to the work that I'm about to be doing in my project??*

**Articles**:
https://www.endo.theclinics.com/article/S0889-8529(19)30091-X/fulltext
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6661119/
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8398465/