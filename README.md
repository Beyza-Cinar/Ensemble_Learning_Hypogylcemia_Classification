# Ensemble Learning for Multihorizon Hypoglycemia Classification

This project evaluates various ML and DL models for the multihorizon hypogylcemia classification in patients with type 1 diabetes using DiaData, a large CGM dataset comprising 2510 subjects. The mdoels were trained unsing univariate data of CGM values. The dataset can be found in https://zenodo.org/records/16874128. The dataset is furhter described in https://doi.org/10.1051/bioconf/202519503001. The four best performing models are further trained with two different ensemble laerning strategies, weighted soft voting and meta-stacking with a logitic regression. 

## Code Organization

The code is organized as follows: 

- Jupyter_Scripts includes all main Jupyter Scripts used for Data reading and preparation, ablation studies, and training the ML models (Random Forest, CatBoost, XGBoost, LightBGM)
- The Plots Folder includes graphic of the main results.
- THe Python_Models folder includes the python files to train the DL models (ResNet, FCN, TCN, TCN+MHAL). For training, a supercomputer was used.
- The Reports folder includes the outputted reports after testing the model performance.
- The utils folder includes neccessary functions to read and preprocess the data.
- The test_subjects.txt file includes the PtIDs of the subjects in the train fold.
