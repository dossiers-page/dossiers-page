---
title: How to reduce heart disease risk? An exploratory Model Analysis on open data
  set
related: true
tags:
- inspiration
- EMA
- exploratory_model_analysis
- machine_learning
---

Age going above 35 is when some of us starts thinking about fitness (and obesity). While there are many points of inspiration, mine are purely on reducing heart disease risk, particularly when it comes as hereditary.  There are many questions I asked myself. Major ones are 
* How do I know if jogging for 5KM everyday is effective? If so, what part of it makes it effective?
* How much sugar and tempting high calorie food I can have, if I run/not run
* What exactly a fitness band would help me? Just measuring heart beat? Sleep?  

Then I came across this course in coursera which talks exactly about these questions and I was very delighted to hear that, most important factor is maintaining higher heart rate for short period of time which would keep you healthy. Even if you spend 10 minutes a day.  Wow!  That is great.  
![Hacking Exercise for Health](/assets/images/hacking_exercise_health_course.jpg). 

While there are studies already available from those professors, we can try something available in the open from [Kaggle](https://www.kaggle.com/sid321axn/heart-statlog-cleveland-hungary-final). 

![Heart Disease Table](/assets/images/heart_disease_table.jpg)
Out of these parameters, what we can control are only the following 

1. Fasting blood sugar
2. Cholesterol
3. Maximum heart rate achieved

In exploratory model analysis, we can take the complete dataset, train the model and check how each of these features determine the heart risk.  Following plot is created after training the model and interpretation of features using SHAP values. Factors on top are more important. Values on right favors heart disease. Values towards blue are lower for a given feature.  
![Shap Values of factors associated to heart disease ](/assets/images/shap_values_heart_data.jpg)

Since fasting blood surar is a boolean value, it is always better to be on lower side. No further check needed. 

A partial dependance plot (PDPPlot) shows the effect of cholesterol 
![Partial dependance plot for cholesterol](/assets/images/partial_dependance_plot_for_cholesterol.jpg)
If we look at the shap plot and pdp, we see that the unit effect to heart disease is 0.05-0.15.  

The most influencing factor in shap plot is the maximum heart rate achieved. 
![Partial Dependance Plot for maximum heart rate achieved](/assets/images/partial_dependance_plot_for_max_heart_rate.jpg)

We could lower the risk from 0.2 to 0.4 just by making sure that our heart rate can reach above 120. This shows that investing in a good fitness watch and doing a HIIT (high intensity interval training),  where we aim to achieve the maximum heart rate in lowest possible time. 

PS: This is only for an academic interest on interpretting what data shows. I am neither  qualified medical practitioner nor an expert on exercises.  I will do a separate post with details on the model building (including the code).

**Acknowledgements**

The dataset is taken from three other research datasets used in different research papers. The Nature article listing heart disease database and names of popular datasets used in various heart disease research is shared below.
https://www.nature.com/articles/s41597-019-0206-3

The data set is consolidated and [made available in kaggle](https://www.kaggle.com/sid321axn/heart-statlog-cleveland-hungary-final) 

Thanks to [this wonderful post in Kaggle](https://www.kaggle.com/sid321axn/stacked-ensemble-for-heart-disease-classification) whch I have used in data cleanup
