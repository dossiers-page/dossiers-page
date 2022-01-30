---
title: 'Teapot: Simplifying statistical tests with a click'
classes: wide
---

For beginners and non-programmers, getting statistical tests done could create headaches.  What if we only want to know if there is a relationship between one of the independent variable and the outcome (and leave the pre-conditions and finding suitable tests by a program) 
This is everything this post is about. 

All that you need is to drag and drop your data in CSV Format (only the first 500 rows will be processed in this demo).  The application does 
* Data Cleanup
* Checks variable types (nominal/interval) 
* Runs default relationship tests 
* Serves the results in a table 
* As a user, you can select one of the independent variables and the application runs one-sided/two-sided statistical tests (if applicable) 


# Screenshots 
![data load ](/assets/images/wine_quality_file_load.jpg)

![data load ](/assets/images/wine_quality_default_tests.jpg)

![data load ](/assets/images/teapot_ar_condition_cleanup.jpg)

![data load ](/assets/images/teapot_ar_condition.jpg)


<iframe src="https://shocking-cheateau-82195.herokuapp.com/" title="statistical tests" width='100%' height='1000' frameBorder="0"  allowtransparency="true"></iframe>


## Tealang: The engine underneath the application 

Underneath the application is [tealang](https://github.com/tea-lang-org/tea-lang) (python library). To know about the library and why it was written, please watch the video below. 
{% include video id="eyoAqNKTjGQ" provider="youtube" %}

## Source code
[Tepot UI](https://github.com/sahyagiri/osm_roads)
