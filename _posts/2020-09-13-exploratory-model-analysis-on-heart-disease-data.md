---
title: Exploratory Model Analysis on Heart Disease Data
tags:
- EMA
- exploratory_model_analysis
- machine_learning
- EDA
related: true
---

Behind the scenes of the "max heart rate achieved" is good for heart.  This is for people who love programming. 
Unlike the traditional style where we do EDA, we start with model building as shown below .  
![exploratory model analysis steps](/assets/images/exploratory_model_analysis_flow.jpg)
The sceptisism from traditional style programmers in ML is that the ensemble or deep learning models are not interpretable.  This post shows how to utilize the power of non-linearity and ensemble model (RandomForest) to study the relationship of heart disease (outcome) from the given data. 
## Imports 

  ``` python

import warnings
warnings.filterwarnings('ignore')
import pandas
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance

import numpy
from scipy import stats
import shap
from pdpbox import pdp, info_plots  # for partial plots
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

```

## Utility Functions 

  ``` python
def get_categorical_variables(data_frame,threshold=0.70, top_n_values=10):
    likely_categorical = []
    for column in data_frame.columns:
        if 1. * data_frame[column].value_counts(normalize=True).head(top_n_values).sum() > threshold:
            likely_categorical.append(column)
    return likely_categorical

def train_model(x,y):
    feature_model = RandomForestClassifier(n_estimators=40, min_samples_leaf=3,
                                                       max_features=0.5,
                                                       n_jobs=-1,
                                                       oob_score=True,max_depth=12,)
    feature_model.fit(x, y)
    return feature_model
def plot_model_interpretations(model):
    explainer = shap.TreeExplainer(model)
    shap_values=explainer.shap_values(x)
    shap.summary_plot(shap_values[1],x)
    
def plot_partial_dependance(x, feature,model):
    base_features = list(x.columns)
    pdp_dist = pdp.pdp_isolate(model=model, dataset=x, model_features=x.columns,
                               feature=feature)
    pdp.pdp_plot(pdp_dist,feature , plot_pts_dist=True)
		
		
```
## Load the data and clean up 

  ``` python
	
data_frame=pandas.read_csv('heart_statlog_cleveland_hungary_final.csv')
categorical_columns=get_categorical_variables(data_frame)
numerical_columns=[column for column in data_frame.columns if column not in categorical_columns]

# remove outliers 
zscore = numpy.abs(stats.zscore(data_frame[numerical_columns]))
data_frame_no_outliers = data_frame[(zscore < 3).all(axis=1)].copy()
data_frame_no_categorical = pandas.get_dummies(data_frame_no_outliers, drop_first=True)
feature_columns=[ i for i in data_frame_no_categorical.columns if i!='heart_disease']
x=data_frame_no_categorical[feature_columns].copy()
y=data_frame_no_categorical.heart_disease.values
model=train_model(x,y)
plot_model_interpretations(model)
		
```

## The output (SHAP Values)  and partial dependance plot for Cholesterol 
![shap values heart disease](/assets/images/shap_initial_incorrect_values_heart_disease.jpg)
  ``` python
plot_partial_dependance(x,'cholesterol',model)
```
![Partial Dependance Plot, Cholesterol](/assets/images/partial_dependance_plot_for_cholesterol_incorrect.jpg)

This tells that, higher the cholesterol, lower the heart failure risk which is counter-intuitive. There is something wrong with the data. 
Let us do a scatterplot to analyse what is the distribution of cholesterol in the data 
  ``` python
plt.figure(figsize=(20,10))
sns.scatterplot(x = 'cholesterol', y = 'age', hue = 'heart_disease', data = data_frame)
```

![Missing Cholesterol Values, Scatterplot](/assets/images/missing_cholesterol_scatterplot.jpg)

Though there are multiple ways to impute, here let us try by training a regression model on known data. 

  ``` python
cholesterol_train_frame=data_frame_no_categorical[data_frame_no_categorical['cholesterol']>0].copy()
cholesterol_prediction=data_frame_no_categorical[data_frame_no_categorical['cholesterol']<=0].copy()
cholesterol_model = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,
                                                   max_features=0.5,
                                                   n_jobs=-1,
                                                   oob_score=True,max_depth=12)
cholesterol_x=cholesterol_train_frame.drop('cholesterol',axis=1)
cholesterol_y=cholesterol_train_frame.cholesterol.values
cholesterol_model.fit(cholesterol_x, cholesterol_y)
cholesterol_prediction['cholesterol']=cholesterol_model.predict(cholesterol_prediction.drop('cholesterol',axis=1))
clean_frame=cholesterol_train_frame.append(cholesterol_prediction)
plt.figure(figsize=(20,10))
sns.scatterplot(x = 'cholesterol', y = 'age', hue = 'heart_disease', data = clean_frame)

```
![Scatterplot for cholesterol, after clean up](/assets/images/cholesterol_scatterplot_clean.jpg)

## Build the model with clean Cholesterol features and plot 
  ``` python
x=clean_frame[feature_columns].copy()
y=clean_frame.heart_disease.values
model=train_model(x,y)
plot_model_interpretations(model)

```

Using the image with explanations for simplicity (in code, only output plot comes) 
![heart disease factors, shap plot](/assets/images/shap_values_heart_data.jpg)

## Partial Dependance Plot for continuous variables/factors
```python
for numerical_column in numerical_columns:
    plot_partial_dependance(x,numerical_column,model)
```
![Heart disease partial dependance plot features 1](/assets/images/continuous_features_plot1.jpg)
![Heart disease partial dependance plot features 2](/assets/images/continuous_features_plot2.jpg)

**Acknowledgements**

The dataset is taken from three other research datasets used in different research papers. The Nature article listing heart disease database and names of popular datasets used in various heart disease research is shared below.
https://www.nature.com/articles/s41597-019-0206-3

The data set is consolidated and [made available in kaggle](https://www.kaggle.com/sid321axn/heart-statlog-cleveland-hungary-final) 

Thanks to [this wonderful post in Kaggle](https://www.kaggle.com/sid321axn/stacked-ensemble-for-heart-disease-classification) whch I have used in data cleanup
