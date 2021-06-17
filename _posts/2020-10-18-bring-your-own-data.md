---
title: Bring your own data
classes: wide
read_time: false
---

One of the reasons of I love Software Engineering is  because, with right user experience; even a layman can start using any software tool.  When it comes to data analysis, for subject experts (doctors, public health workers, bankers and many others), getting the head around R or Python or even automl is bit hard. This blog and the below tool is an experiment on bringing in data analysis without the theory.  This is a very simple automl framework which is running in a server so that you do not need to install any software to try it out. 

Please add your comments at the bottom of the page on the use of such a tool/how this could be made better etc
## How to get started 
You would need a dataset with independant variables and the outcome variables.  Let us say, we are analysing Mushrooms are edible or poisonous. For thtat we collected properties of mushrooms. The concepts of the input table in CSV is shown below .

![mushroom classification](/assets/images/csv_table_explanation_classification.jpg) 

You can proceed with the instructions after preparing such data on your field of interest. 
To try out sample datasets, please use [google dataset search](https://datasetsearch.research.google.com/search?query=health&docid=xvsrhW6jbJriqg%2BEAAAAAA%3D%3D&filters=WyJbXCJmaWxlX2Zvcm1hdF9jbGFzc1wiLFtcIjZcIl1dIl0%3D&property=ZmlsZV9mb3JtYXRfY2xhc3M%3D)

### Animation explaining the steps
![Instructions](/assets/images/how_to_use_the_analysis_portal.gif)

If you are confused on how to interpret the charts below, please have a [look at the previous post](https://dossiers.page/how-to-reduce-heart-disease-risk-an-exploratory-model-analysis-on-open-data-set/)
### Please wait for the below application to load 
<iframe src="https://pythonapps.dossiers.page:7443/" title="EMA" width='100%' height='4260' frameBorder="0"  allowtransparency="true"></iframe>


## What is happening behind the scenes
1. The application downlods 500 rows of the data 
2. Analyses the outcome/target variable and determines the analysis type (Classification/Regression) 
3. Remove constant/highly correlated/identifier/ high in null values 
4. Encodes categorical variables 
5. Create an ensemble model (RandomForest) using the data
6. The relationships found by model is output using SHAP values (Top 10 features)  
7.  Statistical tests between all the control variables and the outcome  (displays up to top 10 values sorted with lowest P-Values) 
8. Plots Partial Dependance Plot of top 3 continuous variables

## Limitations and Caveats 
* The analysis only covers two types (classification and regression)
* The analysis is done only on first 500 rows as it runs on a free server
* If the analysis is classification, the shap plot importance is showing the outcome for one of the outcomes (say [male,female] --> Plots might be for Female) 
* The model is trained only with minimal number of trees (50) 
* Statistical outcomes are based on automated analysis by Tea-Lang (please see the video here for more details) 

## What happens to the data/ does this web site or associated sites store the data?
For academic interests, this web sites captures the column headers of the data.  But the data itself or the analysis results are not stored 
## What if you are interested in more detailed analysis of your data
Please drop a mail to giri@dossiers.page with your initial report from the page 
## Acknoledgements 
1. The application's auto-detection of statistical tests is made possible by (https://tea-lang.org/). Please watch the introduction video by the author of Tea-Lang here (https://www.youtube.com/watch?v=eyoAqNKTjGQ&t=1705s)
2. The Application UI is built using Streamlit (https://www.streamlit.io/)
3. The shap value plots are using shap library (https://github.com/slundberg/shap/)
4. The Plots are done using PDPBox  (https://github.com/SauceCat/PDPbox)
