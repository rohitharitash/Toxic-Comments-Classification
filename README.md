# Toxic-Comments-Classification
Different level of toxic comments were classified from Wikipedia edit comments. Six target class were present in training set. Project contains basic Exploratory data analysis , text mining techniques and XGBoost implementation.

The solution for this problem is divided into three parts. 

First basic level of exploratory data analysis(EDA) is performed. Results from EDA can be utilised to verify the feature selected by the model.

In the second phase, pre-processing of text data is done. 

In the last phase, classification model was trained using XGBoost and some parameters were tuned for performance enhancement.

The models were trained on Microsoft Azure VM running Linux Ubuntu with 32 GB RAM and 4 processing cores. Project was implemented in R.

-- toxicCommentClassifEADFinal.R contain the  exploratory data analysis

-- xgboostClassifFinal has model training and testing.

-- Toxic Comment Classification Report.pdf is the full project report with detailed analysis.
