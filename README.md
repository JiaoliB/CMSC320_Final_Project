# Spotify Machine Learning Analysis
#### Fall 2024 Data Science Project  
#### Jiaoli Bowden, Anjali Paliyam, Jefferson Cheng, Akansha Dave, Amanda T  
# Contributions: 
### Project idea: 
### Dataset Curation and Preprocessing: 
### Data Exploration and Summary Statistics: 
### ML Algorithm Design/Development: 
### ML Algorithm Training and Test Data Analytics: 
### Visualization, Result Analysis, Conclusion: 
### Final Tutorial Report Creation: 
# Introduction
The introduction should motivate your work: what is your topic? What question(s) are you trying to answer with your analysis? Why is answering those questions important?  
Our topic is Spotify music and how the different aspects of the music influence its popularity. We are choosing this dataset because we listen to music and are interested in it. A large majority of people listen to music, and the most popular music streaming app is Spotify, so we thought that we could get the most accurate data from there. The main question that we are trying to answer with our analysis is whether or not a song's popularity is influenced by its musical aspects. By answering this question, we can find certain trends in popuarlity and we would be able to know exactly what aspects propel an artist's song to the top of Spotify. 
# Data Curation
Cite the source(s) of your data. Explain what it is. Transform the data so that it is ready for analysis. For example, set up a database and use SQL to query for data, or organize a pandas DataFrame.  
The source of our data is from HuggingFace dataset library with 114k Spotify songs from 125 different genres. The columns in the dataset are track_id, artists, album_name, track_name, popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acoutsticness, instrumentalness, liveness, valence, tempo, time_signature, and track_genre. 
```python
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
```
```python
df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")
df = df.rename(columns={'Unnamed: 0':"index"})
df.set_index("index", inplace=True)
df
```
# Exploratory Data Analysis (Checkpoint 2)
In order to have usable data, we first have to clean it by checking for NaN values and remove those columns if the data is insignificant (< 1% of the data)
```python
for c in df.columns:
  if df[c].isna().any():
      print("Column", c, "contains", df[c].isna().sum(), "NaN values")
```
This tells us that only 3 columns have 1 NaN value, so we can just remove those.
```python
df.dropna()
```
All of the datatypes of each column already make sense so no change needed (ie words are strings, numbers are ints, decimals are floats, and True/False is bool).  
The main characteristic of the dataset are: 
```python
print("Main Characteristics of the Dataset:")
print(f"Number of Features: {df.shape[1]}")
print(f"Number of Entries: {df.shape[0]}")
print("Feature Information:")
with pd.option_context('display.max_columns', 14):
  print(df.describe(include=[np.number]))
```
We also found out that certain songs are put in multiple times because they have multiple genres associated with it, so we made a different dataframe with unique songs if we aren't looking at any specific genres
```python
df_unique = df.drop_duplicates(subset=["track_name", "album_name"])
```

# Primary Analysis
Based on the results of your exploration, choose a machine learning technique (e.g., classification, regression, clustering, etc.) that will help you answer the questions you posed in the introduction. Explain your reasoning.
### Hypothesis Test #1
If a song is more danceable, then it will be more likely that the song will be popular, since people will be more motivated to dance to it, therefore increasing its popularity.  
H0: Danceability has no affect on the song's popularity  
Ha: Danceability has a positive affect on the song's popularity  
Confidence Level: 95%, so α = 0.05  
We first normalize the data, and then put it through a t-test to test our hypothesis.
```python
df_unique['popularity_normalized'] = (df_unique['popularity'] - df_unique['popularity'].mean()) / df_unique['popularity'].std()
df_unique['danceability_normalized'] = (df_unique['danceability'] - df_unique['danceability'].mean()) / df_unique['danceability'].std()
t_stat, p_value = stats.ttest_rel(df_unique['danceability_normalized'], df_unique['popularity_normalized'])
print("t-statistic:", t_stat)
print("p-value:", p_value)

plt.figure(figsize=(12, 6))
sns.scatterplot(x='danceability', y='popularity', data=df_unique)
plt.title('Scatter Plot of the relationship between danceability and popularity')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.grid(True)
sns.regplot(x='danceability', y='popularity', data=df_unique, scatter=False, color='red')
plt.show()
```
Danceability and Popularity Hypothesis Testing conclusion:  
Since our p-value = 0.9999999999998874, which is very close to 1 and > 0.05, that means we do not have enough evidence to reject the null hypothesis, so we fail to reject the null hypothesis. This means that danceability likely has no affect on popularity, as we failed to reject the null hypothesis.
#### Outlier Detection
Here, we have code to detect outliers for our hypothesis test.
```python
def detect_outliers_iqr(df, column):
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  return df[(df[column] < lower_bound) | (df[column] > upper_bound)]
```
We will use this to create a box-and-whiskers plot to show the outliers in the two columns we tested.
```python
popularity_outliers = detect_outliers_iqr(df_unique, "popularity")
print("Number of outliers in popularity:", len(popularity_outliers))
plt.boxplot(df_unique["popularity"])
plt.show()
```
```python
danceability_outliers = detect_outliers_iqr(df_unique, "danceability")
print("Outliers in danceability:", len(danceability_outliers["track_name"]))
plt.boxplot(df_unique["danceability"])
plt.show()
```
The amount of outliers in proportion to the data is not that much and the value of the outliers is not that far from the rest of the data. Also because we are testing for if danceability affects popularity, every point counts even outliers, so they were not removed for this test.
### Hypothesis Test #2
Pop, indie-pop, and k-pop are all different subcategories of pop music! Since they are all different aspects of pop, is there a signficant difference between the three? We can look at the valence of the different genres!  
H0: There is no difference in valence between the different genres  
Ha: Some genres have more valence than others.  
Confidence Level: 95%, so α = 0.05  
```python
pop_df = df[df['track_genre'] == 'pop']['valence']
indie_pop_df = df[df['track_genre'] == 'indie-pop']['valence']
k_pop_df = df[df['track_genre'] == 'k-pop']['valence']
_, p = f_oneway(pop_df, indie_pop_df, k_pop_df)
p
```
We get a p-value of 2.4*10^-23 which is extremely significant because p is very much lower than α = 0.05. This means that there is likely a difference between the valences of the different genres. We'll explore where they are in the box plots below.
```python
genres = ['pop', 'indie-pop', 'k-pop']
df_selected = df[df['track_genre'].isin(genres)]

sns.boxplot(x='track_genre', y='valence', data=df_selected, palette='Set3')

plt.title('Distribution of Valence by Genre')
plt.xlabel('Genre')
plt.ylabel('Valence')
```
```python
median_valence = df_selected.groupby('track_genre')['valence'].median().reset_index()
median_valence
```
From the graphs and the medians, we can see the order of valuences per genre is k-pop, pop, and the indie-pop from highest to lowest!  
### Hypothesis Test #3
Examining whether track genre is associated with the explicit content of the songs using the Chi Squared test. 
H0: The track's genre is independent of whether it is explicit or not.  
Ha: There is a relationship between the track's genre and its explicitness.  
Confidence Level: 95%, so α = 0.05  
Here we use a contingency table to analyze the data:
```python
table = pd.crosstab(df_unique['track_genre'], df_unique['explicit'])
print(table)
```
Here is a plot showing relationship between the track's genre and the counts of it being explicit or not  
Legend: (blue = not explicit & red = explicit)  
```python
plt.figure(figsize=(20,13))
plt.plot(table.index, table[True], color='red')
plt.plot(table.index, table[False], color='blue')
plt.title("Track Genre vs. Explicitness")
plt.xlabel("Track Genre")
plt.xticks(rotation=90)
plt.ylabel("Number of Tracks")
```
We will conduct a chi-squared test for this hypothesis test: 
```python
statistic, p_value, dof, expected_freq = chi2_contingency(table)
print("p_value:", p_value)
print("chi_squared_value:", statistic)
```
Conclusion of the chi-squared test: Since our p_value is 0.0, it is less than the alpha value of .05, so we reject the null hypothesis. Hence, the genre of the track does have an effect on whether or not it is explicit. A p-value of 0.0 indicates extremely strong evidence against the null hypothesis.
```python

```
# Visualization
Explain the results and insights of your primary analysis with at least one plot. Make sure that every element of the plots are labeled and explained (don’t forget to include a legend!)
```python

```
# Insights and Conclusions
After reading through the project, does an uninformed reader feel informed about the topic? Would a reader who already knew about the topic feel like they learned more about it?
```python

```
