# Wine Quality Data Set

_Author's note: (This data set is available in UCI at https://archive.ics.uci.edu/ml/datasets/Wine+Quality. An excerpt of the problem statement is reproduced here for convenience.)_

**Abstract:** Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests.

#### Data Set Information:

The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.


#### Attribute Information:

Input variables (based on physicochemical tests):

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol

Output variable (based on sensory data):

12. quality (score between 0 and 10)


## Approach for Wine Quality Model Development
We will be developing separate models for the red wine and white wine datasets because (a) red and white wines are produced differently and (b) the UCI website provides two separate datasets. We want to analyze them separately to show any differences in the models and predictions.

### Project Setup:

Our project adopts a folder structure as follows:

<pre>
wine_quality
    |________ datasets
    |________ models
    |________ deployment
</pre>

### Develop an ML Model

Developing an ML model involves a number of steps. Let us adopt the following Machine Learning Pipeline:

1. Sanity Check 
2. EDA/Preprocessing
3. Feature Engineering
4. Model Building
5. Model Saving
6. Model Deployment
7. Model in Production
8. Observe model behaviour
9. Obtain updated datasets
10. Redo steps 1..9 if required

These are described in more detail in the notebook.

### Deployment
The saved model will be deployed in production and made accessible to users for determining estimated prediction of wine quality. A Flask application is developed for this purpose.

Two approaches are described:

1. Deployment on local machine
2. Deployment on Heroku
