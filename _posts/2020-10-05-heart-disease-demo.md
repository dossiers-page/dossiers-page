---
title: Heart Disease Demo
layout: jupyter_archive
read_time: false
---

<div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container">
      



  <div class="cell text_cell">
    
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
<span class="n">warnings</span><span class="o">.</span><span class="n">filterwarnings</span><span class="p">(</span><span class="s2">"ignore"</span><span class="p">)</span>
<span class="c1">#HIDDEN</span>
<span class="k">class</span> <span class="nc">CategoricalPreprocessing</span><span class="p">(</span><span class="n">BaseEstimator</span><span class="p">,</span> <span class="n">TransformerMixin</span><span class="p">):</span>
    <span class="k">def</span> <span class="nf">__get_categorical_variables</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span><span class="n">data_frame</span><span class="p">,</span><span class="n">threshold</span><span class="o">=</span><span class="mf">0.70</span><span class="p">,</span> <span class="n">top_n_values</span><span class="o">=</span><span class="mi">10</span><span class="p">):</span>
        <span class="n">likely_categorical</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">column</span> <span class="ow">in</span> <span class="n">data_frame</span><span class="o">.</span><span class="n">columns</span><span class="p">:</span>
            <span class="k">if</span> <span class="mf">1.</span> <span class="o">*</span> <span class="n">data_frame</span><span class="p">[</span><span class="n">column</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="n">top_n_values</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span> <span class="o">&gt;</span> <span class="n">threshold</span><span class="p">:</span>
                <span class="n">likely_categorical</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">column</span><span class="p">)</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">likely_categorical</span><span class="o">.</span><span class="n">remove</span><span class="p">(</span><span class="s1">'st_depression'</span><span class="p">)</span>
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
<span class="n">feature_model</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">'random_forest_heart_model_v2'</span><span class="p">)</span>
<span class="n">categorical_transform</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">'categorical_transform_v2'</span><span class="p">)</span>
<span class="n">explainer_random_forest</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">'shap_random_forest_explainer_v2'</span><span class="p">)</span>
<span class="n">numerical_options</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">'numerical_options_dictionary_v2'</span><span class="p">)</span>
<span class="n">categorical_options</span><span class="o">=</span><span class="n">joblib</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">'categorical_options_dictionary_v2'</span><span class="p">)</span>


<span class="n">ui_elements</span><span class="o">=</span><span class="p">[]</span>
<span class="n">style</span> <span class="o">=</span> <span class="p">{</span><span class="s1">'description_width'</span><span class="p">:</span> <span class="s1">'initial'</span><span class="p">}</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">numerical_options</span><span class="o">.</span><span class="n">keys</span><span class="p">():</span>
    <span class="n">minimum</span><span class="o">=</span><span class="n">numerical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">'minimum'</span><span class="p">]</span>
    <span class="n">maximum</span><span class="o">=</span><span class="n">numerical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">'maximum'</span><span class="p">]</span>
    <span class="n">default</span><span class="o">=</span><span class="n">numerical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">'default'</span><span class="p">]</span>
    <span class="k">if</span> <span class="n">i</span><span class="o">!=</span><span class="s1">'st_depression'</span><span class="p">:</span>
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
    <span class="n">options</span><span class="o">=</span><span class="n">categorical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">'options'</span><span class="p">],</span>
    <span class="n">value</span><span class="o">=</span><span class="n">categorical_options</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="s1">'default'</span><span class="p">],</span>
    <span class="n">description</span><span class="o">=</span><span class="n">i</span><span class="p">,</span><span class="n">style</span><span class="o">=</span><span class="n">style</span>
    <span class="p">))</span>
<span class="n">interact_calc</span><span class="o">=</span><span class="n">interact</span><span class="o">.</span><span class="n">options</span><span class="p">(</span><span class="n">manual</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">manual_name</span><span class="o">=</span><span class="s2">"Calculate Risk"</span><span class="p">)</span>

<span class="c1">#HIDDEN</span>
<span class="k">def</span> <span class="nf">get_risk_string</span><span class="p">(</span><span class="n">prediction_probability</span><span class="p">):</span>
    <span class="n">y_val</span> <span class="o">=</span> <span class="n">prediction_probability</span><span class="o">*</span> <span class="mi">100</span>
    <span class="n">text_val</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="n">y_val</span><span class="p">,</span> <span class="mi">1</span><span class="p">))</span> <span class="o">+</span> <span class="s2">"% | "</span>

    <span class="c1"># assign a risk group</span>
    <span class="k">if</span> <span class="n">y_val</span> <span class="o">/</span> <span class="mi">100</span> <span class="o">&lt;=</span> <span class="mf">0.275685</span><span class="p">:</span>
        <span class="n">risk_grp</span> <span class="o">=</span> <span class="s1">' low risk '</span>
    <span class="k">elif</span> <span class="n">y_val</span> <span class="o">/</span> <span class="mi">100</span> <span class="o">&lt;=</span> <span class="mf">0.795583</span><span class="p">:</span>
        <span class="n">risk_grp</span> <span class="o">=</span> <span class="s1">' medium risk '</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">risk_grp</span> <span class="o">=</span> <span class="s1">' high risk '</span>
    
    <span class="k">return</span> <span class="n">text_val</span><span class="o">+</span> <span class="n">risk_grp</span>
<span class="k">def</span> <span class="nf">get_current_prediction</span><span class="p">():</span>
    <span class="n">current_values</span><span class="o">=</span><span class="nb">dict</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">element</span> <span class="ow">in</span> <span class="n">ui_elements</span><span class="p">:</span>
        <span class="n">current_values</span><span class="p">[</span><span class="n">element</span><span class="o">.</span><span class="n">description</span><span class="p">]</span><span class="o">=</span><span class="n">element</span><span class="o">.</span><span class="n">value</span>
    <span class="n">feature_row</span><span class="o">=</span><span class="n">categorical_transform</span><span class="o">.</span><span class="n">transform</span><span class="p">(</span><span class="n">pandas</span><span class="o">.</span><span class="n">DataFrame</span><span class="o">.</span><span class="n">from_dict</span><span class="p">([</span><span class="n">current_values</span><span class="p">]))</span>
    <span class="n">feature_row</span><span class="o">=</span><span class="n">feature_row</span><span class="p">[[</span><span class="s1">'age'</span><span class="p">,</span> <span class="s1">'resting_blood_pressure'</span><span class="p">,</span> <span class="s1">'cholesterol'</span><span class="p">,</span>
       <span class="s1">'max_heart_rate_achieved'</span><span class="p">,</span> <span class="s1">'st_depression'</span><span class="p">,</span> <span class="s1">'sex_female'</span><span class="p">,</span> <span class="s1">'sex_male'</span><span class="p">,</span>
       <span class="s1">'chest_pain_type_non-anginal pain'</span><span class="p">,</span> <span class="s1">'chest_pain_type_asymptomatic'</span><span class="p">,</span>
       <span class="s1">'chest_pain_type_atypical angina'</span><span class="p">,</span> <span class="s1">'chest_pain_type_typical angina'</span><span class="p">,</span>
       <span class="s1">'fasting_blood_sugar_0'</span><span class="p">,</span> <span class="s1">'fasting_blood_sugar_1'</span><span class="p">,</span> <span class="s1">'rest_ecg_normal'</span><span class="p">,</span>
       <span class="s1">'rest_ecg_ST-T wave abnormality'</span><span class="p">,</span>
       <span class="s1">'rest_ecg_left ventricular hypertrophy'</span><span class="p">,</span> <span class="s1">'exercise_induced_angina_0'</span><span class="p">,</span>
       <span class="s1">'exercise_induced_angina_1'</span><span class="p">,</span> <span class="s1">'st_slope_flat'</span><span class="p">,</span> <span class="s1">'st_slope_upsloping'</span><span class="p">,</span>
       <span class="s1">'st_slope_downsloping'</span><span class="p">]]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
    
   
    
    <span class="n">shap_values</span> <span class="o">=</span> <span class="n">explainer_random_forest</span><span class="o">.</span><span class="n">shap_values</span><span class="p">(</span><span class="n">feature_row</span><span class="p">)</span>
    
   
    
    
    <span class="n">updated_fnames</span> <span class="o">=</span> <span class="n">feature_row</span><span class="o">.</span><span class="n">T</span><span class="o">.</span><span class="n">reset_index</span><span class="p">()</span>
    <span class="n">updated_fnames</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">'feature'</span><span class="p">,</span> <span class="s1">'value'</span><span class="p">]</span>
    
    
    <span class="n">risk_prefix</span><span class="o">=</span><span class="s1">'&lt;h2&gt; Risk Level :'</span>
    <span class="n">risk_suffix</span><span class="o">=</span><span class="s1">'&lt;/h2&gt;'</span>
    <span class="n">risk_probability</span><span class="o">=</span><span class="n">feature_model</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">feature_row</span><span class="p">)[</span><span class="mi">0</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">risk_string</span><span class="o">=</span><span class="n">get_risk_string</span><span class="p">(</span><span class="n">risk_probability</span><span class="p">)</span>
    <span class="n">risk_widget</span><span class="o">.</span><span class="n">value</span><span class="o">=</span><span class="n">risk_prefix</span><span class="o">+</span><span class="n">risk_string</span><span class="o">+</span><span class="n">risk_suffix</span>
    <span class="n">updated_fnames</span><span class="p">[</span><span class="s1">'shap_original'</span><span class="p">]</span> <span class="o">=</span> <span class="n">pandas</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">shap_values</span><span class="p">[</span><span class="mi">1</span><span class="p">][</span><span class="mi">0</span><span class="p">])</span>
    <span class="n">updated_fnames</span><span class="p">[</span><span class="s1">'shap_abs'</span><span class="p">]</span> <span class="o">=</span> <span class="n">updated_fnames</span><span class="p">[</span><span class="s1">'shap_original'</span><span class="p">]</span><span class="o">.</span><span class="n">abs</span><span class="p">()</span>

    <span class="n">updated_fnames</span><span class="o">=</span><span class="n">updated_fnames</span><span class="p">[</span><span class="n">updated_fnames</span><span class="p">[</span><span class="s1">'value'</span><span class="p">]</span><span class="o">!=</span><span class="mi">0</span><span class="p">]</span>
    
    
    
    <span class="n">plt</span><span class="o">.</span><span class="n">rcParams</span><span class="o">.</span><span class="n">update</span><span class="p">({</span><span class="s1">'font.size'</span><span class="p">:</span> <span class="mi">22</span><span class="p">})</span>
    <span class="n">df1</span><span class="o">=</span><span class="n">pandas</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">updated_fnames</span><span class="p">[</span><span class="s2">"shap_original"</span><span class="p">])</span>
    <span class="n">df1</span><span class="o">.</span><span class="n">index</span><span class="o">=</span><span class="n">updated_fnames</span><span class="o">.</span><span class="n">feature</span>
    <span class="n">df1</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s1">'shap_original'</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">ascending</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
    <span class="n">df1</span><span class="p">[</span><span class="s1">'positive'</span><span class="p">]</span> <span class="o">=</span> <span class="n">df1</span><span class="p">[</span><span class="s1">'shap_original'</span><span class="p">]</span> <span class="o">&gt;</span> <span class="mi">0</span>
    <span class="n">df1</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">kind</span><span class="o">=</span><span class="s1">'barh'</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="n">df1</span><span class="o">.</span><span class="n">positive</span><span class="o">.</span><span class="n">map</span><span class="p">({</span><span class="kc">True</span><span class="p">:</span> <span class="s1">'g'</span><span class="p">,</span> <span class="kc">False</span><span class="p">:</span> <span class="s1">'r'</span><span class="p">}),</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">7</span><span class="p">,),</span><span class="n">legend</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="c1">#HIDDEN</span>
<span class="n">form_item_layout</span> <span class="o">=</span> <span class="n">Layout</span><span class="p">(</span>
    <span class="n">display</span><span class="o">=</span><span class="s1">'flex'</span><span class="p">,</span>
    <span class="n">flex_flow</span><span class="o">=</span><span class="s1">'row'</span><span class="p">,</span>
    <span class="n">justify_content</span><span class="o">=</span><span class="s1">'space-between'</span>
<span class="p">)</span>

<span class="n">form_items</span> <span class="o">=</span> <span class="n">ui_elements</span>

<span class="n">form</span> <span class="o">=</span> <span class="n">Box</span><span class="p">(</span><span class="n">form_items</span><span class="p">,</span> <span class="n">layout</span><span class="o">=</span><span class="n">Layout</span><span class="p">(</span>
    <span class="n">display</span><span class="o">=</span><span class="s1">'flex'</span><span class="p">,</span>
    <span class="n">flex_flow</span><span class="o">=</span><span class="s1">'column'</span><span class="p">,</span>
   
    <span class="n">align_items</span><span class="o">=</span><span class="s1">'stretch'</span>
   
<span class="p">))</span>


<span class="n">box_layout</span> <span class="o">=</span> <span class="n">Layout</span><span class="p">(</span><span class="n">display</span><span class="o">=</span><span class="s1">'flex'</span><span class="p">,</span>
                    <span class="n">flex_flow</span><span class="o">=</span><span class="s1">'column'</span>
                    <span class="p">)</span>

<span class="n">left_box</span> <span class="o">=</span> <span class="n">VBox</span><span class="p">(</span><span class="n">ui_elements</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">5</span><span class="p">])</span>
<span class="n">right_box</span> <span class="o">=</span> <span class="n">VBox</span><span class="p">(</span><span class="n">ui_elements</span><span class="p">[</span><span class="mi">5</span><span class="p">:])</span>
<span class="n">control_layout</span><span class="o">=</span><span class="n">VBox</span><span class="p">([</span><span class="n">left_box</span><span class="p">,</span><span class="n">right_box</span><span class="p">],</span><span class="n">layout</span><span class="o">=</span><span class="n">box_layout</span><span class="p">)</span>

<span class="n">risk_string</span><span class="o">=</span><span class="s2">"&lt;h2&gt;Risk Level&lt;/h2&gt;"</span>

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
<span class="n">risk_plot</span><span class="o">.</span><span class="n">widget</span><span class="o">.</span><span class="n">layout</span><span class="o">=</span><span class="n">Layout</span><span class="p">(</span><span class="n">display</span><span class="o">=</span><span class="s1">'flex'</span><span class="p">,</span>
                    <span class="n">flex_flow</span><span class="o">=</span><span class="s1">'column'</span><span class="p">,</span><span class="n">width</span><span class="o">=</span><span class="s1">'50%'</span>
                    <span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    



  <div class="output_subarea output_widget_view ">
    
  <div class="p-Widget p-Panel jupyter-widgets widget-container widget-box widget-vbox" style="display: flex; flex-flow: column;"><div class="p-Widget p-Panel jupyter-widgets widget-container widget-box widget-vbox"><div class="p-Widget jupyter-widgets widget-inline-hbox widget-slider widget-hslider"><label class="widget-label" title="age" style="width: initial;">age</label><div class="slider-container"><div class="ui-slider ui-corner-all ui-widget ui-widget-content slider ui-slider-horizontal"><span tabindex="0" class="ui-slider-handle ui-corner-all ui-state-default" style="left: 56.5217%;"></span></div></div><div class="widget-readout" contenteditable="true" style="">57</div></div><div class="p-Widget jupyter-widgets widget-inline-hbox widget-slider widget-hslider"><label class="widget-label" title="resting_blood_pressure" style="width: initial;">resting_blood_pressure</label><div class="slider-container"><div class="ui-slider ui-corner-all ui-widget ui-widget-content slider ui-slider-horizontal"><span tabindex="0" class="ui-slider-handle ui-corner-all ui-state-default" style="left: 50%;"></span></div></div><div class="widget-readout" contenteditable="true" style="">130</div></div><div class="p-Widget jupyter-widgets widget-inline-hbox widget-slider widget-hslider"><label class="widget-label" title="cholesterol" style="width: initial;">cholesterol</label><div class="slider-container"><div class="ui-slider ui-corner-all ui-widget ui-widget-content slider ui-slider-horizontal"><span tabindex="0" class="ui-slider-handle ui-corner-all ui-state-default" style="left: 55.8233%;"></span></div></div><div class="widget-readout" contenteditable="true" style="">239</div></div><div class="p-Widget jupyter-widgets widget-inline-hbox widget-slider widget-hslider"><label class="widget-label" title="max_heart_rate_achieved" style="width: initial;">max_heart_rate_achieved</label><div class="slider-container"><div class="ui-slider ui-corner-all ui-widget ui-widget-content slider ui-slider-horizontal"><span tabindex="0" class="ui-slider-handle ui-corner-all ui-state-default" style="left: 53.5433%;"></span></div></div><div class="widget-readout" contenteditable="true" style="">135</div></div><div class="p-Widget jupyter-widgets widget-inline-hbox widget-slider widget-hslider"><label class="widget-label" title="st_depression" style="width: initial;">st_depression</label><div class="slider-container"><div class="ui-slider ui-corner-all ui-widget ui-widget-content slider ui-slider-horizontal"><span tabindex="0" class="ui-slider-handle ui-corner-all ui-state-default" style="left: 37.5%;"></span></div></div><div class="widget-readout" contenteditable="true" style="">1.40</div></div></div><div class="p-Widget p-Panel jupyter-widgets widget-container widget-box widget-vbox"><div class="p-Widget jupyter-widgets widget-inline-hbox widget-dropdown"><label class="widget-label" title="sex" for="721f91ff-267c-46f9-a09e-50792b1d2dee" style="width: initial;">sex</label><select id="721f91ff-267c-46f9-a09e-50792b1d2dee"><option data-value="female" value="female">female</option><option data-value="male" value="male">male</option></select></div><div class="p-Widget jupyter-widgets widget-inline-hbox widget-dropdown"><label class="widget-label" title="chest_pain_type" for="addec937-0177-457c-9c29-881385e9a449" style="width: initial;">chest_pain_type</label><select id="addec937-0177-457c-9c29-881385e9a449"><option data-value="non-anginal%20pain" value="non-anginal pain">non-anginal&nbsp;pain</option><option data-value="asymptomatic" value="asymptomatic">asymptomatic</option><option data-value="atypical%20angina" value="atypical angina">atypical&nbsp;angina</option><option data-value="typical%20angina" value="typical angina">typical&nbsp;angina</option></select></div><div class="p-Widget jupyter-widgets widget-inline-hbox widget-dropdown"><label class="widget-label" title="fasting_blood_sugar" for="0d8767c6-464a-429a-807e-f3d63ebb4326" style="width: initial;">fasting_blood_sugar</label><select id="0d8767c6-464a-429a-807e-f3d63ebb4326"><option data-value="0" value="0">0</option><option data-value="1" value="1">1</option></select></div><div class="p-Widget jupyter-widgets widget-inline-hbox widget-dropdown"><label class="widget-label" title="rest_ecg" for="3722d41b-94b2-4ecf-859e-78abe7324348" style="width: initial;">rest_ecg</label><select id="3722d41b-94b2-4ecf-859e-78abe7324348"><option data-value="normal" value="normal">normal</option><option data-value="ST-T%20wave%20abnormality" value="ST-T wave abnormality">ST-T&nbsp;wave&nbsp;abnormality</option><option data-value="left%20ventricular%20hypertrophy" value="left ventricular hypertrophy">left&nbsp;ventricular&nbsp;hypertrophy</option></select></div><div class="p-Widget jupyter-widgets widget-inline-hbox widget-dropdown"><label class="widget-label" title="exercise_induced_angina" for="75f1ff48-2074-4096-a91a-b76adeb0d509" style="width: initial;">exercise_induced_angina</label><select id="75f1ff48-2074-4096-a91a-b76adeb0d509"><option data-value="0" value="0">0</option><option data-value="1" value="1">1</option></select></div><div class="p-Widget jupyter-widgets widget-inline-hbox widget-dropdown"><label class="widget-label" title="st_slope" for="d95c1cfc-63e6-450c-9601-948b5ecde94c" style="width: initial;">st_slope</label><select id="d95c1cfc-63e6-450c-9601-948b5ecde94c"><option data-value="flat" value="flat">flat</option><option data-value="upsloping" value="upsloping">upsloping</option><option data-value="downsloping" value="downsloping">downsloping</option></select></div></div></div><div class="p-Widget p-Panel jupyter-widgets widget-container widget-box"><div class="p-Widget jupyter-widgets widget-inline-hbox widget-html"><label class="widget-label" title="" style="display: none;"></label><div class="widget-html-content"><h2>Risk Level</h2></div></div></div><div class="p-Widget p-Panel jupyter-widgets widget-container widget-box widget-vbox widget-interact" style="display: flex; flex-flow: column; width: 50%;"><button class="p-Widget jupyter-widgets jupyter-button widget-button" title="">Calculate Risk</button><div class="p-Widget p-Panel jupyter-widgets widget-output"><div class="p-Widget jp-OutputArea"></div></div></div></div>

</div>

<div class="output_area">

    



  <div class="output_subarea output_widget_view ">
    
  </div>

</div>

<div class="output_area">

    



  <div class="output_subarea output_widget_view ">
    
  </div>

</div>

</div>
</div>

  </div>

  

  <div class="
      cell border-box-sizing code_cell rendered">
    <div class="input">


</div>

  </div>



<!-- Loads nbinteract package -->
<script src="https://unpkg.com/nbinteract-core" async=""></script>
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
