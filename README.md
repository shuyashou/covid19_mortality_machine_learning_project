# Covid-19 Mortality Project

### Project Summary:
In this project, we aim to investigate the intricate relationship between different factors and the mortality rate of COVID-19 by employing a range of machine-learning techniques. Our objective is to develop predictive models that not only enhance our understanding of risk factors but also aid public health strategies in mitigating the impact of pandemic.

### Project Links:
#### GitHub Repo: https://github.com/shuyashou/cs475_final_project
#### Web Interface: https://cs475project-c17babadc414.herokuapp.com/
#### Complete Project Report: https://github.com/shuyashou/cs475_final_project/blob/main/jupyter/final_report.ipynb

### Directory Organization:
- **main branch**
```
/main
│
├── /jupyter                 
│   ├── final_report.ipynb
│
├── /script                 
│   ├── download.py  
│   └── preprocessing.py
│
└── /python                 
    ├── logreg_findings.py           
    ├── randomforest_findings.py
    └── ...
```
- **app branch**
```
/app
│
├── /app                  
│   ├── /templates
│   │   ├── index.html
│   │
│   │
│   ├── __init__.py
│   ├── app.py
│   ├── logreg.py
│   ├── randomforest.py
│   ├── model_lr.pkl
│   ├── model_rf.pkl
│   └── ...
│
├── Procfile                             
├── requirements.txt         
└── runtime.txt             

```


> [!NOTE]
> **Deliverables Adjustments:**
> 
> - **"Must complete #2":** Implementation and evaluation of logistic regression, Random Forest, SVM, XGBoost, and AdaBoost models in predicting COVID-19 outcomes.
>   - After discussing with our TA (Thomas speaking to the TA), we felt that the right direction for our project was to refine our strongest models (Logistic Regression and Random Forest) and further analyze them regarding our findings. Originally, we had some initial approaches to SVM, XGBoost, and AdaBoost but they severely lacked any key findings, comprehensive analysis or good baseline performance evaluations. We felt that we should prioritize quality over quantity when it came to our models where we'd fare better focusing on improving our existing models vs. opting to simply introduce new models without the proper ML pipeline. One of the key weaknesses of the paper that introduced our idea of using ML for Covid was this shortcoming where they did not consider a large dataset, comprehensive evluation metrics or extensive optimization techniques.
> 
> 3. **"Must complete #3":** Data visualization of COVID-19 trends over time using matplotlib and seaborn libraries.
> 
> 4. **"Must complete #4":** Documentation of model performance metrics and findings in a detailed report.


### Uncompleted Deliverables
1. "Expect to complete #2": we decided to use an existing implementation for our SVM
2. "Would like to accomplish #2": Collaboration with healthcare professionals (e.g., Johns Hopkins Medicine) for model validation and refinement alongside medical expertise that may be of significance for our findings. In addition to healthcare professionals, outreach to the authors that inspired our project topic (cited as [1]) to discuss our findings and methodologies.
3. "Would like to accomplish #3": Submit research findings in a peer-reviewed journal and/or present them at relevant conferences and forums to share insights and contribute to the collective understanding of COVID-19 dynamics.

### Completed Deliverables
1. "Must complete #1": Development of a data preprocessing pipeline tailored to the CDC COVID-19 dataset.
    - Within the "Data Cleaning and Preprocessing" section of final_report.ipynb
2. "Must complete #2": Implementation and evaluation of logistic regression, Random Forest, SVM, XGBoost, and AdaBoost models in predicting COVID-19 outcomes.
    - Within the "Logistic Regression" and "Random Forest" section of final_report.ipynb
3. "Must complete #3": Analysis of our various models, and their results in identifying the key contributors to mortality due to Covid-19.
    - Within the latter sections of "Logistic Regression" and "Random Forest", and "Insights and Recommendations"
4. "Expect to accomplish #1": Explore techniques such as SHAP (SHapley Additive exPlanations) values, partial dependence plots, and permutation importance to gain deeper understanding of feature contributions.
    - With the graphics available in sections of "Logistic Regression" and "Random Forest"
5. "Expect to accomplish #2": Enhancement of model accuracy through hyperparameter optimization and cross-validation techniques.
    - Shown in latter sections of "Logistic Regression" and "Random Forest"
6. "Expect to accomplish #3": Creation of a user-friendly interface for real-time prediction of COVID-19 outcomes based on patient data. This user-friendly interface could be accessible via website.
    - [Model Prediction Interface](https://cs475project-c17babadc414.herokuapp.com/)
    - Note if clicking the above hyperlink doesn't work, the URL is as follows: <https://cs475project-c17babadc414.herokuapp.com/>

### Additional Deliverables
1. 
2. 
