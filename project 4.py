#!/usr/bin/env python
# coding: utf-8

# # Import library for read the dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("advertising.csv")


# In[3]:


df


# # Data Analysis

# In[4]:


df.info()


# In[5]:


print(df.columns)


# In[6]:


column_names = ['TV', 'Radio', 'Newspaper', 'Sales']

for column in column_names:
    column_data = df[column]
    print(f"Column: {column}")
    print(column_data)
    print()


# In[8]:


df.isnull()


# In[9]:


df.isnull().sum()


# In[11]:


df.count()


# In[13]:


print(df.describe)


# In[14]:


print(df.shape)


# In[15]:


print(df.dtypes)


# In[16]:


filtered_data = df[(df['Radio'] >= 3.7) & (df['Radio'] <= 10.8)]

print(filtered_data)


# In[17]:


filtered_data = df[(df['TV'] >= 180) & (df['TV'] <= 230)]

print(filtered_data)


# In[18]:


filtered_data = df[(df['Newspaper'] >= 40) & (df['Newspaper'] <= 60)]

print(filtered_data)


# In[20]:


filtered_data = df[(df['Sales'] >= 12) & (df['Sales'] <= 15)]

print(filtered_data)


# # Data Visualization

# In[21]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[22]:


df.hist(bins=10, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[24]:


# Define the colors based on conditions
colors = ['red' if length >= 120 else 'yellow' for length in df['TV']]

# Scatter plot with different colors
plt.scatter(df['TV'], df['Radio'], c=colors)
plt.xlabel('TV')
plt.ylabel('Radio')
plt.title('TV vs Radio')

# Show the plot
plt.show()


# In[29]:


# Define the colors based on conditions
colors = ['red' if length >= 45 else 'yellow' for length in df['Newspaper']]

# Scatter plot with different colors
plt.scatter(df['Newspaper'], df['Radio'], c=colors)
plt.xlabel('Newspaper')
plt.ylabel('Radio')
plt.title('Newspaper vs Radio')

# Show the plot
plt.show()


# In[31]:


# Define the colors based on conditions
colors = ['red' if length >= 15 else 'yellow' for length in df['Sales']]

# Scatter plot with different colors
plt.scatter(df['Sales'], df['TV'], c=colors)
plt.xlabel('Sales')
plt.ylabel('TV')
plt.title('Sales vs TV')

# Show the plot
plt.show()


# In[32]:


# Select the columns for correlation
columns = ['TV', 'Radio', 'Newspaper', 'Sales']

# Calculate the coefficient matrix
correlation_matrix = df[columns].corr()

# Display the coefficient matrix
print(correlation_matrix)

# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[34]:


sns.countplot(x='TV', data=df, )
plt.show()


# In[35]:


#check Outliers
sns.heatmap(df.isnull(), yticklabels=False, annot=True)


# In[36]:


sns.scatterplot(x='TV', y='Sales',
                hue='TV', data=df, )
 
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
 
plt.show()


# In[37]:


sns.scatterplot(x='Radio', y='Sales',
                hue='Radio', data=df, )
 
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
 
plt.show()


# In[38]:


sns.scatterplot(x='Newspaper', y='Sales',
                hue='Newspaper', data=df, )
 
# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)
 
plt.show()


# In[40]:


sns.pairplot(df, hue='TV', height=2)


# In[46]:


plt.figure(figsize=(10, 10))

# Creating box plots for 'TV', 'Radio', and 'Newspaper' against 'Sales'
plt.subplot(2, 2, 1)
sns.boxplot(x='TV', y='Sales', data=df)

plt.subplot(2, 2, 2)
sns.boxplot(x='Radio', y='Sales', data=df)

plt.subplot(2, 2, 3)
sns.boxplot(x='Newspaper', y='Sales', data=df)

plt.tight_layout()
plt.show()


# In[42]:


plot = sns.FacetGrid(df, hue="TV")
plot.map(sns.distplot, "Sales").add_legend()

plot = sns.FacetGrid(df, hue="Radio")
plot.map(sns.distplot, "Sales").add_legend()

plot = sns.FacetGrid(df, hue="Newspaper")
plot.map(sns.distplot, "Sales").add_legend()

plt.show()


# In[47]:


# Distribution of a numerical column
sns.histplot(df['Sales'], kde=True)
plt.title('Distribution of Sales')
plt.show()


# # Machine Learning 
# # Linear Regression  Models:-
# 

# In[61]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming your dataset is stored in a DataFrame named 'df'

# Splitting the data into features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting on the test set
y_pred = lr.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Calculating accuracy
threshold = 0.1  # Define your threshold value here
accurate_predictions = (abs(y_test - y_pred) <= threshold).sum()
total_predictions = len(y_test)
accuracy = accurate_predictions / total_predictions

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)
print("Accuracy:", accuracy)


# # RandomForestRegressor model

# In[68]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Assuming your dataset is stored in a DataFrame named 'df'

# Splitting the data into features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the RandomForestRegressor model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predicting on the test set
y_pred = rf.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)


# In[ ]:





# In[ ]:




