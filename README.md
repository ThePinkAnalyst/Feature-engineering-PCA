# Principal Component Analysis onG loabsl Digital Competitiveness Dataset

## Project Overview
The primary objective of this project was to perform Principal Component Analysis (PCA) on a dataset related to digital competitiveness across various countries. The dataset initially consisted of 24 variables, including economic indicators, technological adoption metrics, and education and infrastructure data. The goal was to reduce the dimensionality of the dataset while retaining the most significant components that explain the majority of the variance in the data.

## Dataset Description
The original dataset contained the following variables:
### Country and Demographics:
Country,
Country Key, and
Year.

### Skills and Research:
Foreign highly skilled personnel,
Digital/Technological skills,
Total expenditure on R&D (%), and
Scientific research legislation.

### Technology and Infrastructure:
Mobile broadband subscribers,
EParticipation,
Use of big data and analytics, and
Health infrastructure.

### Economy and Labor:
Skilled labor,
Gross Domestic Product (GDP),
GDP (PPP) per capita,
Exports of goods growth,
Government budget surplus/deficit (%), and
Tax evasion.

### Social Indicators:
Pension funding,
Protectionism,
Equal opportunity, and
Disposable income.

### Corporate and Energy Consumption:
Listed domestic companies,
Image abroad or branding,
Digital transformation in companies, and
Total final energy consumption per capita.


## Tools & Technologies Used
Programming Languages: Python.
Libraries/Frameworks: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib.
Software/Platforms: Jupyter Notebook

## Methodology
### Exploratory Data Analysis
The exploratory data analysis (EDA) phase included:

Correlation Analysis:
A correlation heatmap was generated to visualize the relationships between the numerical variables in the dataset.

Pairplots and Distribution Plots:
Pairplots were used to examine the relationships between pairs of variables, helping identify patterns and potential correlations. Distribution plots were utilized to visualize the distribution of each variable, offering insights into their spread and central tendencies.

Histograms:
Histograms were plotted for each variable to understand their frequency distribution and detect any potential outliers or skewness.

### Principal Component Analysis (PCA)
Fitting PCA:
PCA was applied to the dataset to reduce the dimensionality. The explained variance ratio for each principal component was calculated and visualized.

Optimal Number of Components:
The elbow method was employed to determine the optimal number of components to retain. This involved plotting the cumulative explained variance ratio against the number of components. An explained variance ratio of 95% was considered as statistically significant.
The optimal number of components was determined to be 14, explaining the majority of the variance in the data.

Final PCA Application:
PCA was performed using the optimal 14 components. The resulting components were ranked by their importance, and the most significant features were selected.

Component Analysis:
The selected components were analyzed to identify the most influential variables, which were then used to create a reduced dataset for further analysis.

## Findings
After performing PCA and using the elbow method to determine the optimal number of principal components, the dataset was reduced to the following 14 principal components:
-Foreign highly skilled personnel
-Total expenditure on R&D (%)
-Scientific research legislation
-Mobile broadband subscribers
-EParticipation
-Use of big data and analytics
-Total public exp. on education per student
-Health infrastructure
-Skilled labor
-Gross Domestic Product (GDP)
-GDP (PPP) per capita
-Exports of goods growth
-Government budget surplus/deficit (%)
-Tax evasion
