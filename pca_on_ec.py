
import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

datawithnan= pd.read_csv('endometrial.csv')

data_with_gene_id = datawithnan.dropna() #remove missing values
data = data_with_gene_id.iloc[:,1:]
#print(data.head())

scaled_data = preprocessing.scale(data.T) # we use transpose because the scale function expects the samples as rows 

pca = PCA(n_components = 10) # create a PCA object
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum()) #cumulative variance
#print(scaled_data)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()









