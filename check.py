# for collection of data i have used dataset from kaggle for  heart failure prediction dataset 
import pandas as pd
# 1 visualization of data 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("heart.csv")   # making a data frame variable for dataset 
df.head() # displaying few values 

# for there if any missing values .
df.isnull().sum()

# removing any duplicate values 
df.drop_duplicates(inplace=True)

#for summary of database with datatypes 
df.info()

#for insights in data 
df.describe()



# Histogram for numerical features
df.hist(figsize=(12, 8), bins=30)
plt.show()

#now findin the relation between the features with corelation
plt.figure(figsize=(10, 6))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
''' if non numeric number (string ) eixst it will give error (ValueError: could not convert string to float: 'M')
so thst is why check or filter all the numeric value using select_dtypes()
 '''

df_numeric = df.select_dtypes(include=["number"])
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# now target varaiable analysis means heart desease occurance 
sns.countplot(x="HeartDisease", data=df, palette="viridis")
#sns.countplot(x="HeartDisease", hue="HeartDisease", data=df, palette="viridis", legend=False) # to avoid future errors 
""" as we palette is resposible for resposible for color we can also write  """
#sns.countplot(x="HeartDisease", data=df, color="blue") # any color but all the column will be of same color 
plt.title("Heart Disease Count ")
plt.show()