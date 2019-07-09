# Wine-Classification-using-Linear-Discriminant-Analysis
Classification of Wines by implementation Linear Discriminant Analysis (LDA)

### Reason of Choosing Dataset ?

- For the purpose of applying Linear Discriminant Analysis I'm choosing this dataset.


- However, this Statistical models are not prepared to use for production environment.

### Source:

- Original Owners : 

    - Forina, M. et al, PARVUS - An Extendible Package for Data Exploration, Classification and Correlation. Institute of Pharmaceutical and Food Analysis and Technologies, Via Brigata Salerno, 16147 Genoa, Italy. 


- Donor : 
    - Stefan Aeberhard, email: stefan '@' coral.cs.jcu.edu.au


- Download :
    - https://archive.ics.uci.edu/ml/datasets/wine



### Data Set Information :

These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. 

I think that the initial data set had around 30 variables, but for some reason I only have the 13 dimensional version. I had a list of what the 30 or so variables were, but a.) I lost it, and b.), I would not know which 13 variables are included in the set. 

The attributes are (dontated by Riccardo Leardi, riclea '@' anchem.unige.it ) 

- Alcohol
- Malic acid 
- Ash 
- Alcalinity of ash 
- Magnesium 
- Total phenols 
- Flavanoids 
- Nonflavanoid phenols 
- Proanthocyanins 
- Color intensity 
- Hue 
- OD280/OD315 of diluted wines 
- Proline 

In a classification context, this is a well posed problem with "well behaved" class structures. A good data set for first testing of a new classifier, but not very challenging.

### Attribute Information :

- All attributes are continuous 


- No statistics available, but suggest to standardise variables for certain uses (e.g. for us with classifiers which are NOT scale invariant) 


- NOTE: 1st attribute is class identifier (1-3)

### Objective :

- The goal is to make some predictive models on a wine dataset by implementing Linear Discriminant Analysis, and reviewing some exploratory and modelling techiniques.
