# %% [markdown]
# ## SOUTH WALES POLICE DATA

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import folium 
from folium import plugins
from folium.plugins import MarkerCluster

plt.rcParams['figure.figsize']=[10,8]
sns.set_style('darkgrid')

# %%
#create dataframe crime
crime=pd.DataFrame()
loc='data'
directory=os.fsencode(loc)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filepath=loc + '/' + filename
    df=pd.read_csv(filepath)
    crime=pd.concat([crime,df], ignore_index=True)
crime.head()

# %%
print("Number of rows in dataset is: ", crime.shape[0])
print("Number of columns in dataset is: ", crime.shape[1])

# %% [markdown]
# ## Data Information

# %%
crime.info()

# %%
crime.shape

# %%
sns.heatmap(crime.isnull(),yticklabels=False,cbar=True,cmap='ocean');

# %%
crime.drop(columns=['Context','Crime ID','Last outcome category'],inplace=True)

# %%
crime.dropna(inplace=True)

# %%
crime.shape

# %%
##Visulizing

# %%
crime.Month.value_counts()

# %%
crime.Month.value_counts().plot.bar()

# %%
crime['Crime type'].value_counts()

# %%
sns.countplot(data=crime,y='Crime type',order=crime['Crime type'].value_counts().index)

# %%
sns.countplot(data=crime,y='Location',order=crime['Location'].value_counts().index[:10])

# %%
sns.countplot(data=crime,y='LSOA name',order=crime['LSOA name'].value_counts().index[:10])

# %%
m=folium.Map([51.57,-3.45],zoom_start=10)
dfmatrix=crime[['Latitude','Longitude']].values
plugins.HeatMap(dfmatrix,radius=15).add_to(m)
for index, row in crime.iterrows():
    folium.CircleMarker([row['Latitude'],row['Longitude']],radius=3,popup=row['Crime type'],
                        fill_color='blue').add_to(m)
m

# %%
lsoa=pd.read_csv('LSOA.csv')
lsoa.head()

# %%
lsoa.drop(columns=['LA Code (2018 boundaries)','LA name (2018 boundaries)','LA Code (2021 boundaries)',
                   'LA name (2021 boundaries)'], inplace=True)

# %%
lsoa.head()

# %%
crime['LSOA code'].unique()
lsoa_code=crime['LSOA code'].unique()

# %%
lsoa=lsoa[lsoa['LSOA Code'].isin(lsoa_code)]

# %%
lsoa.set_index('LSOA Code', inplace=True)
lsoa.head()

# %%
lsoa_crime = pd.DataFrame(crime['LSOA code'].value_counts())
lsoa_crime.sample(n=20)

# %%
lsoa['Crime']=lsoa_crime['LSOA code']
lsoa.head()

# %%
lsoa[~lsoa.index.str.contains('E')]

# %%
#linear regression
"""
x will be the median age and y will be the crime.
do ols summary and then scatter plot with prediction line
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

# %%
lsoa.describe()

# %%
y = lsoa['Crime']
x1 = lsoa[['Median Age']]

# %%
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

# %%
plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1,yhat, lw=4, c='orange', label='regression line')
plt.xlabel('Median Age', fontsize = 10)
plt.ylabel('Crime', fontsize = 10)
plt.show()

# %%
x = sm.add_constant(x1)
reg_lin = sm.OLS(y,x)
results_lin = reg_lin.fit()

plt.scatter(x1,y,color = 'C0')
y_hat = x1*results_lin.params[1]+results_lin.params[0]
plt.plot(x1,y_hat,lw=2.5,color='C8')
plt.xlabel('Median Age', fontsize = 20)
plt.ylabel('Crime', fontsize = 20)
plt.show()

# %%
lsoa.describe()

# %%
x_scaled = lsoa.copy()
x_scaled = x_scaled.drop(['Crime'],axis=1)

# %%
x_scaled

# %%
sns.distplot(lsoa['Crime'], kde=True)

# %% [markdown]
# ## SUSSEX POLICE DATA

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import folium 
from folium import plugins
from folium.plugins import MarkerCluster

plt.rcParams['figure.figsize']=[10,8]
sns.set_style('darkgrid')

# %%
#create dataframe crime
crime=pd.DataFrame()
loc='data 1'
directory=os.fsencode(loc)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filepath=loc + '/' + filename
    df=pd.read_csv(filepath)
    crime=pd.concat([crime,df], ignore_index=True)
crime.head()

# %%
print("Number of rows in dataset is: ", crime.shape[0])
print("Number of columns in dataset is: ", crime.shape[1])

# %% [markdown]
# # Data information

# %%
crime.info()

# %%
crime.shape

# %%
sns.heatmap(crime.isnull(),yticklabels=False,cbar=True,cmap='Blues');

# %%
crime.drop(columns=['Context','Crime ID','Last outcome category'],inplace=True)

# %%
crime.dropna(inplace=True)

# %%
crime.shape

# %%
##Visulization

# %%
crime.Month.value_counts()

# %%
crime.Month.value_counts().plot.bar()

# %%
crime['Crime type'].value_counts()

# %%
sns.countplot(data=crime,y='Crime type',order=crime['Crime type'].value_counts().index)

# %%
sns.countplot(data=crime,y='Location',order=crime['Location'].value_counts().index[:10])

# %%
sns.countplot(data=crime,y='LSOA name',order=crime['LSOA name'].value_counts().index[:10])

# %%
m=folium.Map([51.57,-3.45],zoom_start=10)
dfmatrix=crime[['Latitude','Longitude']].values
plugins.HeatMap(dfmatrix,radius=15).add_to(m)
for index, row in crime.iterrows():
    folium.CircleMarker([row['Latitude'],row['Longitude']],radius=3,popup=row['Crime type'],
                        fill_color='blue').add_to(m)
m

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

# %%
lsoa=pd.read_csv('LSOA.csv')
lsoa.head()

# %%
lsoa.drop(columns=['LA Code (2018 boundaries)','LA name (2018 boundaries)','LA Code (2021 boundaries)',
                   'LA name (2021 boundaries)'], inplace=True)

# %%
lsoa.head()

# %%
crime['LSOA code'].unique()
lsoa_code=crime['LSOA code'].unique()

# %%
lsoa=lsoa[lsoa['LSOA Code'].isin(lsoa_code)]

# %%
lsoa.set_index('LSOA Code', inplace=True)
lsoa.head()

# %%
lsoa_crime = pd.DataFrame(crime['LSOA code'].value_counts())
lsoa_crime.sample(n=20)

# %%
lsoa['Crime']=lsoa_crime['LSOA code']
lsoa.head()

# %%
lsoa[~lsoa.index.str.contains('E')]

# %%
lsoa.describe()

# %%
y = lsoa['Crime']
x1 = lsoa[['Median Age']]

# %%
x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

# %%
plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1,yhat, lw=4, c='orange', label='regression line')
plt.xlabel('Median Age', fontsize = 10)
plt.ylabel('Crime', fontsize = 10)
plt.show()

# %%
x = sm.add_constant(x1)
reg_lin = sm.OLS(y,x)
results_lin = reg_lin.fit()

plt.scatter(x1,y,color = 'C0')
y_hat = x1*results_lin.params[1]+results_lin.params[0]
plt.plot(x1,y_hat,lw=2.5,color='C8')
plt.xlabel('Median Age', fontsize = 20)
plt.ylabel('Crime', fontsize = 20)
plt.show()

# %%
x_scaled = lsoa.copy()
x_scaled = x_scaled.drop(['Crime'],axis=1)

# %%
x_scaled

# %%
sns.distplot(lsoa['Crime'], kde=True)

# %%



