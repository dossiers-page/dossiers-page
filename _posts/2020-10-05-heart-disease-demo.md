---
title: 'Heart Disease Demo: How much risk for a single patient has'
layout: jupyter_single
read_time: false
comments: true
share: true
related: true
tags:
- EMA
- exploratory_model_analysis
- xai
- explainable_ai
---

The previous post explained, overall what factors affect health of the heart. This is the global interpretability of machine learning.  A better tool for a subject matter expert to try out is on the boundary conditions.  Here, he/she can put values which even confuses the experts; and see how well the model behaves. 
This is called **local interpretability**
Here is an example of local interpretablity and how the model behaves well and eratically, given the input conditions.  

The method used here is the use of [SHAPELY  values](https://en.wikipedia.org/wiki/Shapley_value). To get an idea how this works, think of a game where each team member contributes to the final score.

## A note on the parameters used in the demo
1. Age: Patients Age in years
2.  Resting blood pressure : Level of blood pressure at resting mode in mm/HG (Systoloc)
3.  Cholestrol: Serum cholestrol in mg/dl 
4.  Maximum Heart Rate Achieved: Heart rate achieved while doing a threadmill test or exercise
5.  ST_Depression/oldpeak: Exercise induced ST-depression in comparison with the state of rest 
6. Sex: Gender of patient (The data had only male and female) 
7.  Chest Pain Type: Type of chest pain experienced by patient
8.  Fasting blood sugar: Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false
9. Resting ecg: Result of electrocardiogram while at rest
10. Exercise angina: Angina induced by exercise 0 depicting NO 1 depicting Yes
11.  ST slope: ST segment measured in terms of slope during peak exercise
##  Try it out
(If you are loading this for first time, click on show widgets below, to load the application. Best viewed in bigger screen) 
<div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container">
      



  <div class="cell text_cell">
    <button class="js-nbinteract-widget">
      Loading widgets...
    </button>
  </div>




  

  <div class="nbinteract-hide_in
      cell border-box-sizing code_cell rendered">
    <div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#HIDDEN</span>
<span class="kn">import</span> <span class="nn">pandas</span>
<span class="kn">import</span> <span class="nn">shap</span>
<span class="kn">import</span> <span class="nn">joblib</span>
<span class="kn">import</span> <span class="nn">ipywidgets</span> <span class="k">as</span> <span class="nn">widgets</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.patches</span> <span class="k">as</span> <span class="nn">mpatches</span>
<span class="kn">from</span> <span class="nn">sklearn.base</span> <span class="kn">import</span> <span class="n">TransformerMixin</span><span class="p">,</span> <span class="n">BaseEstimator</span>
<span class="kn">from</span> <span class="nn">pandas</span> <span class="kn">import</span> <span class="n">Categorical</span><span class="p">,</span> <span class="n">get_dummies</span>
<span class="kn">import</span> <span class="nn">time</span>
<span class="kn">from</span> <span class="nn">ipywidgets</span> <span class="kn">import</span> <span class="n">interact</span><span class="p">,</span> <span class="n">interact_manual</span><span class="p">,</span><span class="n">interactive</span>
<span class="kn">from</span> <span class="nn">ipywidgets</span> <span class="kn">import</span> <span class="n">Layout</span><span class="p">,</span> <span class="n">Button</span><span class="p">,</span> <span class="n">Box</span><span class="p">,</span> <span class="n">VBox</span>
<span class="kn">from</span> <span class="nn">ipywidgets</span> <span class="kn">import</span> <span class="n">Button</span><span class="p">,</span> <span class="n">HBox</span><span class="p">,</span> <span class="n">VBox</span>
<span class="kn">from</span> <span class="nn">ipywidgets</span> <span class="kn">import</span> <span class="n">Layout</span><span class="p">,</span> <span class="n">Button</span><span class="p">,</span> <span class="n">Box</span><span class="p">,</span> <span class="n">FloatText</span><span class="p">,</span> <span class="n">Textarea</span><span class="p">,</span> <span class="n">Dropdown</span><span class="p">,</span> <span class="n">Label</span><span class="p">,</span> <span class="n">IntSlider</span>
<span class="kn">import</span> <span class="nn">warnings</span>
<span class="n">warnings</span><span class="o">.</span><span class="n">filterwarnings</span><span class="p">(</span><span class="s2">&quot;ignore&quot;</span><span class="p">)</span>
<span class="c1">#HIDDEN</span>
<span class="k">class</span> <span class="nc">CategoricalPreprocessing</span><span class="p">(</span><span class="n">BaseEstimator</span><span class="p">,</span> <span class="n">TransformerMixin</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__get_categorical_variables</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span><span class="n">data_frame</span><span class="p">,</span><span class="n">threshold</span><span class="o">=</span><span class="mf">0.70</span><span class="p">,</span> <span class="n">top_n_values</span><span class="o">=</span><span class="mi">10</span><span class="p">):</span>
        <span class="n">likely_categorical</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">column</span> <span class="ow">in</span> <span class="n">data_frame</span><span class="o">.</span><span class="n">columns</span><span class="p">:</span>
            <span class="k">if</span> <span class="mf">1.</span> <span class="o">*</span> <span class="n">data_frame</span><span class="p">[</span><span class="n">column</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="n">top_n_values</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">&gt;</span> <span class="n">threshold</span><span class="p">:</span>
                <span class="n">likely_categorical</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">column</span><span class="p">)</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">likely_categorical</span><span class="o">.</span><span class="n">remove</span><span class="p">(</span><span class="s1">&#39;st_depression&#39;</span><span class="p">)</span>
        <span class="k">except</span><span class="p">:</span>
            <span class="k">pass</span> 
        <span class="k">return</span> <span class="n">likely_categorical</span>
    
    <span class="k">def</span> <span class="nf">fit</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">attribute_names</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">__get_categorical_variables</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>
        <span class="n">cats</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="k">for</span> <span class="n">column</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">attribute_names</span><span class="p">:</span>
            <span class="n">cats</span><span class="p">[</span><span class="n">column</span><span class="p">]</span> <span class="o">=</span> <span class="n">X</span><span class="p">[</span><span class="n">column</span><span class="p">]</span><span class="o">.</span><span class="n">unique</span><span class="p">()</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">categoricals</span> <span class="o">=</span> <span class="n">cats</span>
        <span class="k">return</span> <span class="bp">self</span>

    <span class="k">def</span> <span class="nf">transform</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
        <span class="n">df</span> <span class="o">=</span> <span class="n">X</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
        <span class="k">for</span> <span class="n">column</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">attribute_names</span><span class="p">:</span>
            <span class="n">df</span><span class="p">[</span><span class="n">column</span><span class="p">]</span> <span class="o">=</span> <span class="n">Categorical</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="n">column</span><span class="p">],</span> <span class="n">categories</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">categoricals</span><span class="p">[</span><span class="n">column</span><span class="p">])</span>
        <span class="n">new_df</span> <span class="o">=</span> <span class="n">get_dummies</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">drop_first</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
        <span class="c1"># in case we need them later</span>
        <span class="k">return</span> <span class="n">new_df</span>
<span class="c1">#HIDDEN</span>
<span class="n">feature_model</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">&#39;random_forest_heart_model_v2&#39;</span><span class="p">)</span>
<span class="n">categorical_transform</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">&#39;categorical_transform_v2&#39;</span><span class="p">)</span>
<span class="n">explainer_random_forest</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">&#39;shap_random_forest_explainer_v2&#39;</span><span class="p">)</span>
<span class="n">numerical_options</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">&#39;numerical_options_dictionary_v2&#39;</span><span class="p">)</span>
<span class="n">categorical_options</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">&#39;categorical_options_dictionary_v2&#39;</span><span class="p">)</span>


<span class="n">ui_elements</span><span class="o">=</span><span class="p">[]</span>
<span class="n">style</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;description_width&#39;</span><span class="p">:</span> <span class="s1">&#39;initial&#39;</span><span class="p">}</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">numerical_options</span><span class="o">.</span><span class="n">keys</span><span class="p">():</span>
    <span class="n">minimum</span><span class="o">=</span><span class="n">numerical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">&#39;minimum&#39;</span><span class="p">]</span>
    <span class="n">maximum</span><span class="o">=</span><span class="n">numerical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">&#39;maximum&#39;</span><span class="p">]</span>
    <span class="n">default</span><span class="o">=</span><span class="n">numerical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">&#39;default&#39;</span><span class="p">]</span>
    <span class="k">if</span> <span class="n">i</span><span class="o">!=</span><span class="s1">&#39;st_depression&#39;</span><span class="p">:</span>
        <span class="n">ui_elements</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">widgets</span><span class="o">.</span><span class="n">IntSlider</span><span class="p">(</span>
        <span class="n">value</span><span class="o">=</span><span class="n">default</span><span class="p">,</span>
        <span class="nb">min</span><span class="o">=</span><span class="n">minimum</span><span class="p">,</span>
        <span class="nb">max</span><span class="o">=</span><span class="n">maximum</span><span class="p">,</span>
        <span class="n">step</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
        <span class="n">description</span><span class="o">=</span><span class="n">i</span><span class="p">,</span><span class="n">style</span><span class="o">=</span><span class="n">style</span><span class="p">)</span>
                      <span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">ui_elements</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">widgets</span><span class="o">.</span><span class="n">FloatSlider</span><span class="p">(</span>
        <span class="n">value</span><span class="o">=</span><span class="n">default</span><span class="p">,</span>
        <span class="nb">min</span><span class="o">=</span><span class="n">minimum</span><span class="p">,</span>
        <span class="nb">max</span><span class="o">=</span><span class="n">maximum</span><span class="p">,</span>
        <span class="n">step</span><span class="o">=.</span><span class="mi">5</span><span class="p">,</span>
        <span class="n">description</span><span class="o">=</span><span class="n">i</span><span class="p">,</span><span class="n">style</span><span class="o">=</span><span class="n">style</span><span class="p">)</span>
                      <span class="p">)</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">categorical_options</span><span class="o">.</span><span class="n">keys</span><span class="p">():</span>
    <span class="n">ui_elements</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">widgets</span><span class="o">.</span><span class="n">Dropdown</span><span class="p">(</span>
    <span class="n">options</span><span class="o">=</span><span class="n">categorical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">&#39;options&#39;</span><span class="p">],</span>
    <span class="n">value</span><span class="o">=</span><span class="n">categorical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">&#39;default&#39;</span><span class="p">],</span>
    <span class="n">description</span><span class="o">=</span><span class="n">i</span><span class="p">,</span><span class="n">style</span><span class="o">=</span><span class="n">style</span>
    <span class="p">))</span>
<span class="n">interact_calc</span><span class="o">=</span><span class="n">interact</span><span class="o">.</span><span class="n">options</span><span class="p">(</span><span class="n">manual</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">manual_name</span><span class="o">=</span><span class="s2">&quot;Calculate Risk&quot;</span><span class="p">)</span>

<span class="c1">#HIDDEN</span>
<span class="k">def</span> <span class="nf">get_risk_string</span><span class="p">(</span><span class="n">prediction_probability</span><span class="p">):</span>
    <span class="n">y_val</span> <span class="o">=</span> <span class="n">prediction_probability</span><span class="o">*</span> <span class="mi">100</span>
    <span class="n">text_val</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="n">y_val</span><span class="p">,</span> <span class="mi">1</span><span class="p">))</span> <span class="o">+</span> <span class="s2">&quot;% | &quot;</span>

    <span class="c1"># assign a risk group</span>
    <span class="k">if</span> <span class="n">y_val</span> <span class="o">/</span> <span class="mi">100</span> <span class="o">&lt;=</span> <span class="mf">0.275685</span><span class="p">:</span>
        <span class="n">risk_grp</span> <span class="o">=</span> <span class="s1">&#39; low risk &#39;</span>
    <span class="k">elif</span> <span class="n">y_val</span> <span class="o">/</span> <span class="mi">100</span> <span class="o">&lt;=</span> <span class="mf">0.795583</span><span class="p">:</span>
        <span class="n">risk_grp</span> <span class="o">=</span> <span class="s1">&#39; medium risk &#39;</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">risk_grp</span> <span class="o">=</span> <span class="s1">&#39; high risk &#39;</span>
    
    <span class="k">return</span> <span class="n">text_val</span><span class="o">+</span> <span class="n">risk_grp</span>
<span class="k">def</span> <span class="nf">get_current_prediction</span><span class="p">():</span>
    <span class="n">current_values</span><span class="o">=</span><span class="nb">dict</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">element</span> <span class="ow">in</span> <span class="n">ui_elements</span><span class="p">:</span>
        <span class="n">current_values</span><span class="p">[</span><span class="n">element</span><span class="o">.</span><span class="n">description</span><span class="p">]</span><span class="o">=</span><span class="n">element</span><span class="o">.</span><span class="n">value</span>
    <span class="n">feature_row</span><span class="o">=</span><span class="n">categorical_transform</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">pandas</span><span class="o">.</span><span class="n">DataFrame</span><span class="o">.</span><span class="n">from_dict</span><span class="p">([</span><span class="n">current_values</span><span class="p">]))</span>
    <span class="n">feature_row</span><span class="o">=</span><span class="n">feature_row</span><span class="p">[[</span><span class="s1">&#39;age&#39;</span><span class="p">,</span> <span class="s1">&#39;resting_blood_pressure&#39;</span><span class="p">,</span> <span class="s1">&#39;cholesterol&#39;</span><span class="p">,</span>
       <span class="s1">&#39;max_heart_rate_achieved&#39;</span><span class="p">,</span> <span class="s1">&#39;st_depression&#39;</span><span class="p">,</span> <span class="s1">&#39;sex_female&#39;</span><span class="p">,</span> <span class="s1">&#39;sex_male&#39;</span><span class="p">,</span>
       <span class="s1">&#39;chest_pain_type_non-anginal pain&#39;</span><span class="p">,</span> <span class="s1">&#39;chest_pain_type_asymptomatic&#39;</span><span class="p">,</span>
       <span class="s1">&#39;chest_pain_type_atypical angina&#39;</span><span class="p">,</span> <span class="s1">&#39;chest_pain_type_typical angina&#39;</span><span class="p">,</span>
       <span class="s1">&#39;fasting_blood_sugar_0&#39;</span><span class="p">,</span> <span class="s1">&#39;fasting_blood_sugar_1&#39;</span><span class="p">,</span> <span class="s1">&#39;rest_ecg_normal&#39;</span><span class="p">,</span>
       <span class="s1">&#39;rest_ecg_ST-T wave abnormality&#39;</span><span class="p">,</span>
       <span class="s1">&#39;rest_ecg_left ventricular hypertrophy&#39;</span><span class="p">,</span> <span class="s1">&#39;exercise_induced_angina_0&#39;</span><span class="p">,</span>
       <span class="s1">&#39;exercise_induced_angina_1&#39;</span><span class="p">,</span> <span class="s1">&#39;st_slope_flat&#39;</span><span class="p">,</span> <span class="s1">&#39;st_slope_upsloping&#39;</span><span class="p">,</span>
       <span class="s1">&#39;st_slope_downsloping&#39;</span><span class="p">]]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
    
   
    
    <span class="n">shap_values</span> <span class="o">=</span> <span class="n">explainer_random_forest</span><span class="o">.</span><span class="n">shap_values</span><span class="p">(</span><span class="n">feature_row</span><span class="p">)</span>
    
   
    
    
    <span class="n">updated_fnames</span> <span class="o">=</span> <span class="n">feature_row</span><span class="o">.</span><span class="n">T</span><span class="o">.</span><span class="n">reset_index</span><span class="p">()</span>
    <span class="n">updated_fnames</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;feature&#39;</span><span class="p">,</span> <span class="s1">&#39;value&#39;</span><span class="p">]</span>
    
    
    <span class="n">risk_prefix</span><span class="o">=</span><span class="s1">&#39;&lt;h2&gt; Risk Level :&#39;</span>
    <span class="n">risk_suffix</span><span class="o">=</span><span class="s1">&#39;&lt;/h2&gt;&#39;</span>
    <span class="n">risk_probability</span><span class="o">=</span><span class="n">feature_model</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">feature_row</span><span class="p">)[</span><span class="mi">0</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">risk_string</span><span class="o">=</span><span class="n">get_risk_string</span><span class="p">(</span><span class="n">risk_probability</span><span class="p">)</span>
    <span class="n">risk_widget</span><span class="o">.</span><span class="n">value</span><span class="o">=</span><span class="n">risk_prefix</span><span class="o">+</span><span class="n">risk_string</span><span class="o">+</span><span class="n">risk_suffix</span>
    <span class="n">updated_fnames</span><span class="p">[</span><span class="s1">&#39;shap_original&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">pandas</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">shap_values</span><span class="p">[</span><span class="mi">1</span><span class="p">][</span><span class="mi">0</span><span class="p">])</span>
    <span class="n">updated_fnames</span><span class="p">[</span><span class="s1">&#39;shap_abs&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">updated_fnames</span><span class="p">[</span><span class="s1">&#39;shap_original&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">abs</span><span class="p">()</span>

    <span class="n">updated_fnames</span><span class="o">=</span><span class="n">updated_fnames</span><span class="p">[</span><span class="n">updated_fnames</span><span class="p">[</span><span class="s1">&#39;value&#39;</span><span class="p">]</span><span class="o">!=</span><span class="mi">0</span><span class="p">]</span>
    
    
    
    <span class="n">plt</span><span class="o">.</span><span class="n">rcParams</span><span class="o">.</span><span class="n">update</span><span class="p">({</span><span class="s1">&#39;font.size&#39;</span><span class="p">:</span> <span class="mi">30</span><span class="p">})</span>
    <span class="n">df1</span><span class="o">=</span><span class="n">pandas</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">updated_fnames</span><span class="p">[</span><span class="s2">&quot;shap_original&quot;</span><span class="p">])</span>
    <span class="n">df1</span><span class="o">.</span><span class="n">index</span><span class="o">=</span><span class="n">updated_fnames</span><span class="o">.</span><span class="n">feature</span>
    <span class="n">df1</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s1">&#39;shap_original&#39;</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">ascending</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">df1</span><span class="p">[</span><span class="s1">&#39;positive&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df1</span><span class="p">[</span><span class="s1">&#39;shap_original&#39;</span><span class="p">]</span> <span class="o">&gt;</span> <span class="mi">0</span>
    <span class="n">df1</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">kind</span><span class="o">=</span><span class="s1">&#39;barh&#39;</span><span class="p">,</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">7</span><span class="p">,),</span><span class="n">legend</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="c1">#HIDDEN</span>
<span class="n">form_item_layout</span> <span class="o">=</span> <span class="n">Layout</span><span class="p">(</span>
    <span class="n">display</span><span class="o">=</span><span class="s1">&#39;flex&#39;</span><span class="p">,</span>
    <span class="n">flex_flow</span><span class="o">=</span><span class="s1">&#39;row&#39;</span><span class="p">,</span>
    <span class="n">justify_content</span><span class="o">=</span><span class="s1">&#39;space-between&#39;</span>
<span class="p">)</span>

<span class="n">form_items</span> <span class="o">=</span> <span class="n">ui_elements</span>

<span class="n">form</span> <span class="o">=</span> <span class="n">Box</span><span class="p">(</span><span class="n">form_items</span><span class="p">,</span> <span class="n">layout</span><span class="o">=</span><span class="n">Layout</span><span class="p">(</span>
    <span class="n">display</span><span class="o">=</span><span class="s1">&#39;flex&#39;</span><span class="p">,</span>
    <span class="n">flex_flow</span><span class="o">=</span><span class="s1">&#39;column&#39;</span><span class="p">,</span>
   
    <span class="n">align_items</span><span class="o">=</span><span class="s1">&#39;stretch&#39;</span>
   
<span class="p">))</span>


<span class="n">box_layout</span> <span class="o">=</span> <span class="n">Layout</span><span class="p">(</span><span class="n">display</span><span class="o">=</span><span class="s1">&#39;flex&#39;</span><span class="p">,</span>
                    <span class="n">flex_flow</span><span class="o">=</span><span class="s1">&#39;column&#39;</span>
                    <span class="p">)</span>

<span class="n">left_box</span> <span class="o">=</span> <span class="n">VBox</span><span class="p">(</span><span class="n">ui_elements</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">5</span><span class="p">])</span>
<span class="n">right_box</span> <span class="o">=</span> <span class="n">VBox</span><span class="p">(</span><span class="n">ui_elements</span><span class="p">[</span><span class="mi">5</span><span class="p">:])</span>
<span class="n">control_layout</span><span class="o">=</span><span class="n">VBox</span><span class="p">([</span><span class="n">left_box</span><span class="p">,</span><span class="n">right_box</span><span class="p">],</span><span class="n">layout</span><span class="o">=</span><span class="n">box_layout</span><span class="p">)</span>

<span class="n">risk_string</span><span class="o">=</span><span class="s2">&quot;&lt;h2&gt;Risk Level&lt;/h2&gt;&quot;</span>

<span class="n">risk_widget</span><span class="o">=</span><span class="n">widgets</span><span class="o">.</span><span class="n">HTML</span><span class="p">(</span><span class="n">value</span><span class="o">=</span><span class="n">risk_string</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

  </div>

  

  <div class="nbinteract-hide_in
      cell border-box-sizing code_cell rendered">
    <div class="input">

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#HIDDEN</span>
<span class="n">display</span><span class="p">(</span><span class="n">control_layout</span><span class="p">)</span>
<span class="n">display</span><span class="p">(</span><span class="n">Box</span><span class="p">(</span><span class="n">children</span><span class="o">=</span><span class="p">[</span><span class="n">risk_widget</span><span class="p">]))</span>
<span class="n">risk_plot</span><span class="o">=</span><span class="n">interact_calc</span><span class="p">(</span><span class="n">get_current_prediction</span><span class="p">)</span>

<span class="n">risk_plot</span><span class="o">.</span><span class="n">widget</span><span class="o">.</span><span class="n">children</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">style</span><span class="o">.</span><span class="n">button_color</span> <span class="o">=</span> <span class="s1">&#39;lightblue&#39;</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    



  <div class="output_subarea output_widget_view ">
    <button class="js-nbinteract-widget">
      Loading widgets...
    </button>
  </div>

</div>

<div class="output_area">

    



  <div class="output_subarea output_widget_view ">
    <button class="js-nbinteract-widget">
      Loading widgets...
    </button>
  </div>

</div>

<div class="output_area">

    



  <div class="output_subarea output_widget_view ">
    <button class="js-nbinteract-widget">
      Loading widgets...
    </button>
  </div>

</div>

</div>
</div>

  </div>



<!-- Loads nbinteract package -->
<script src="https://unpkg.com/nbinteract-core" async></script>
<script>
  (function setupNbinteract() {
    // If NbInteract hasn't loaded, wait one second and try again
    if (window.NbInteract === undefined) {
      setTimeout(setupNbinteract, 1000)
      return
    }

    var interact = new window.NbInteract({
      spec: 'sahyagiri/heart_risk_local/master',
      baseUrl: 'https://mybinder.org',
      provider: 'gh',
    })
    interact.prepare()

    window.interact = interact
  })()
</script>
    </div>
  </div>

Please note: This is running in free servers and you may need to wait for it to load correctly.

## Below image shows how the interactoin (below) is supposed to render 
![Local interpretation of heart disease](/assets/images/local_explanation_example_1.jpg)

The things on positive axis contribute positively to the heart risk and things on negative axis contribute towards good heart health. 

## Below image shows if we reduce the risk 

![Local interpretation of heart disease](/assets/images/local_explanation_example_2_healthy.jpg)

Here we can see if a person is healthy at 57 years, how the lab results and the corresponding risk would look like 

## An anomaly with cholesterol levels

![Local interpretation of heart disease](/assets/images/local_explanation_example_3_anomaly.jpg)

** Here the model thinks that high cholesterol is good for health.    **

This is why it is always important to give interactive widgets to the subject matter experts (here a doctor) to try it out first than giving a set of charts. 
The next iteration in a model building would be to look at the data and see what pattern emerges which makes this/ train a different model/ tune the model parameter to look for specific patterns.



**Acknowledgements**

The dataset is taken from three other research datasets used in different research papers. The Nature article listing heart disease database and names of popular datasets used in various heart disease research is shared below.
https://www.nature.com/articles/s41597-019-0206-3

The data set is consolidated and [made available in kaggle](https://www.kaggle.com/sid321axn/heart-statlog-cleveland-hungary-final) 

Thanks to [this wonderful post in Kaggle](https://www.kaggle.com/sid321axn/stacked-ensemble-for-heart-disease-classification) whch I have used in data cleanup
