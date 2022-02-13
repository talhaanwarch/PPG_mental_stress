# Title  
Machine Learning Based Real-Time Diagnosis of Mental Stress Using Photoplethysmography  

# Abstract  
Mental stress is a natural response to life activities. However, acute and prolonged stress may cause psychological and heart diseases. Heart rate variability (HRV) is considered an indicator of mental stress and physical fitness. The standard way of obtaining HRV is using electrocardiography (ECG) as the time interval between two consecutive R-peaks. ECG signal is collected by attaching electrodes on different locations of the body, which need a proper clinical setup and is costly as well; therefore, it is not feasible to monitor stress with ECG. Photoplethysmography (PPG) is considered an alternative for mental stress detection using pulse rate variability (PRV), the time interval between two successive peaks of PPG. This study aims to diagnose daily life stress using low-cost portable PPG devices instead of lab trials and expensive devices. Data is collected from 27 subjects both in rest and in stressed conditions in daily life routine. Thirty-six time domain, frequency domain, and non-linear features are extracted from PRV. Multiple machine learning classifiers are used to classify these features. Recursive feature elimination, student t-test and genetic algorithm are used to select these features. An accuracy of 72% is achieved using stratified leave out cross-validation using K-Nearest Neighbor, and it increased up to 81% using a genetic algorithm. Once the model is trained with the best features selected with the genetic algorithm, we used the trained weights for the real-time prediction of mental stress. The results show that using a low-cost device; stress can be diagnosed in real life. The proposed method enable the regular monitoring of stress in short time that help to control the occurrence of psychological and cardiovascular diseases.  

# Results

|     Classifiers               |     Accuracy    |     GS Accuracy    |     T-test    |     GA accuracy    |     RFE Accuracy    |
|-------------------------------|-----------------|--------------------|---------------|--------------------|---------------------|
|     Logistic Regression       |     59%         |     63%            |     72%       |     74%            |     72%             |
|     Support Vector Machine    |     69%         |     72%            |     76%       |     72%            |     78%             |
|     K-Nearest Neighbor        |     65%         |     72%            |     74%       |     81%            |     76%             |
|     Decision Tree             |     54%         |     66%            |     72%       |     74%            |     76%             |
|     Random Forest             |     61%         |     70%            |     72%       |     72%            |     75%             |  

# Note:  
Dataset is publicly available. In case you use it, please cite following paper.  

# Reference  
```  
@article{anwar2022,
author = {Anwar, Talha and Zakir, Seemab},
title = {Machine Learning Based Real-Time Diagnosis of Mental Stress Using Photoplethysmography},
year = {2022},
month = {4},
volume = {55},
pages = {154--167},
journal = {Journal of Biomimetics, Biomaterials and Biomedical Engineering},
}
```
