# TMDB_Box_Office_prediction

## The solution contains the following parts
1. EDA of features
2. Visualization of relationship between features and revenue
3. Transformation of features
4. Training with random forest

## EDA
The solution examine 2 types of features: numerical and categorical
Numerical features are chosen based on value of correlation between them and revenue
Categorical features are chosen based on effects on binary encoding, features counting and one-hot encoding
Codes related: EDA.py

## Visualization
Diagrams of effects of different preprocessing to features are stored in /EDA

## Transformation of data for training
Features are handled with null value and 0 values
Then the original dataset is transformed to cleaned dataset ready for training
Codes related: create_csv.py

## Training
Machine learning algorithm used in this project is Random Forest due to the abundance of categorical data.
Training and prediction is performed in same file to mitigate the need of output the model
Codes related: train_N_test.py, training.ipynb (used on google Colab to access Pycaret for automatic model selection and tuning)
