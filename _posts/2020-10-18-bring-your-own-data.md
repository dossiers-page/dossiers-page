---
title: Bring your own data
---

One reason I love Software Engineering is that, with right user experience; even a layman can start using any software tool.  When it comes to data analysis, for subject experts (doctors, public health workers, bankers and many others), getting the head around R or Python or even automl is bit hard. This blog and the below tool is an experiment on bringing in data analysis without the theory. 

## How to get started (animation below) 
![Instructions](/assets/images/how_to_use_the_analysis_portal.gif)
<iframe src="http://localhost:8501/" title="EDA" width='100%' height='4260' frameBorder="0"  allowtransparency="true"></iframe>

## What is happening behind the scenes
1. The application downloads your data
2. Analyses the outcome/target variable and determines the analysis type (Classification/Regression) 
3. Encodes categorical variables 
4. Checks if all these variables are relevant for the analysis 
5. Create an ensemble model (RandomForest) using the data
6. The relationships found by model is output using SHAP values (Top 10 features)  
7. Statistical tests between all the control variables and the outcome is run (using Tea-Lang) 
8. Plots Partial Dependance Plot of top 3 continuous variables

## Limitations and Caveats 
* The analysis only covers two types (classification and regression)
* The analysis is done only on first 250 rows as it runs on a free server
* If the analysis is classification, the shap plot importance is showing the outcome for one of the outcomes (say [male,female] --> Plots might be for Female) 
* The model is trained only with minimal number of trees (50) 
* Statistical outcomes are based on automated analysis by Tea-Lang (please see the video here for more details) 

## Acknoledgements 
1. The application's auto-detection and variable selection are part of [Mlbox](https://mlbox.readthedocs.io/en/latest/) 
2. The application's auto-detection of statistical tests is made possible by [Tea-Lang](https://tea-lang.org/)
3. The dataset in demo is  from three other research datasets used in different research papers. The Nature article listing heart disease database and names of popular datasets used in various heart disease research is shared below.
https://www.nature.com/articles/s41597-019-0206-3 . The data set is consolidated and [made available in kaggle](https://www.kaggle.com/sid321axn/heart-statlog-cleveland-hungary-final)
