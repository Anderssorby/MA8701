import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import dump, load

import pandas as pd
import numpy as np
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LassoCV
from sklearn.utils import parallel_backend
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mdf = 50
price_ratio = True

train_size = 50000
test_size = 50000
# get data from csv files
data = pd.read_csv('train.csv', usecols=['price', 'description', 'item_seq_number', 'category_name', 'deal_probability'])
# Extract a nicer amount of data
data = data.sample(train_size+test_size, random_state = 8701)
# Replace missing values, create missing price variable
data['missingprice'] = data['price'].isna()
data['item_seq_number'].fillna(0, inplace=True)
data['description'].fillna(" ", inplace=True)
data['desclength'] = data['description'].str.len()

# Split data into train and test data
traindata = data[:train_size]
testdata = data[train_size:]
data=0

trainprice, trainmissingprice, traindesc, traindesclength, trainitemnum, train_category, trainY = traindata['price'], traindata['missingprice'], traindata['description'], traindata['desclength'], traindata['item_seq_number'], traindata['category_name'], traindata['deal_probability']
traincategory = pd.get_dummies(pd.Categorical(train_category), prefix = 'category')
testprice, testmissingprice, testdesc, testdesclength, testitemnum, testcategory, testY = testdata['price'], testdata['missingprice'], testdata['description'], testdata['desclength'], testdata['item_seq_number'], testdata['category_name'], testdata['deal_probability']
testcategory = pd.get_dummies(pd.Categorical(testcategory), prefix = 'category')

if price_ratio:
    df = pd.concat([trainprice,train_category], axis=1)
    df.dropna(subset=['price'])
    categorymeans = np.asmatrix(df.groupby('category_name').mean()) #Calculate mean price within each category, NaNs excluded
    df = 0
else:
    trainprice.fillna(0, inplace=True)
    testprice.fillna(0, inplace=True)

#Convert dataframes to numpy arrays
trainprice = np.transpose(np.asmatrix(trainprice))
testprice = np.transpose(np.asmatrix(testprice))
trainmissingprice = np.transpose(np.asmatrix(trainmissingprice))
testmissingprice = np.transpose(np.asmatrix(testmissingprice))
trainitemnum = np.transpose(np.asmatrix(trainitemnum))
testitemnum = np.transpose(np.asmatrix(testitemnum))
traincategory = np.asmatrix(traincategory)
testcategory = np.asmatrix(testcategory)
traindesclength = np.transpose(np.asmatrix(traindesclength))
testdesclength = np.transpose(np.asmatrix(testdesclength))

if price_ratio:
    for i in range(train_size): #Divide price by category mean from training set, NaN price ratios are set to 1
        cat = np.nonzero(traincategory[i])[1][0]
        trainprice[i] = 1 if math.isnan(trainprice[i]) else trainprice[i]/categorymeans[cat]

    for i in range(test_size): #Same for test set, using means from training set
        cat = np.nonzero(testcategory[i])[1][0]
        testprice[i] = 1 if math.isnan(testprice[i]) else testprice[i]/categorymeans[cat]
else:
    trainpricecut = np.percentile(trainprice,80) #Cuts prices larger than the 80th percentile
    for i in range(trainprice.shape[0]): #Requires the test set to be at least as large as the training set
        if trainprice[i,0] > trainpricecut:
            trainprice[i,0] = trainpricecut
        if testprice[i,0] > trainpricecut:
            testprice[i,0] = trainpricecut

## Get "bag of words" transformation of the data
vec = TfidfVectorizer(ngram_range=(1, 1),
                      min_df=mdf,
                      max_df=0.9,
                      lowercase=True,
                      strip_accents='unicode',
                      sublinear_tf=True)

trainX = vec.fit_transform(traindesc)
testX = vec.transform(testdesc)

print(trainX.shape)

#plt.figure()
#plt.hist(trainprice)
#plt.show()
#plt.plot(trainprice, trainY, 'b.')
#plt.show()

#Add all numeric features
trainX = np.append(trainX.todense(),trainprice,1)
trainX = np.append(trainX,trainitemnum,1)
trainX = np.append(trainX,traindesclength,1)
testX = np.append(testX.todense(),testprice,1)
testX = np.append(testX,testitemnum,1)
testX = np.append(testX,testdesclength,1)

#Standardize numeric features
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

#Add categorical features
trainX = np.append(trainX,trainmissingprice,1)
trainX = np.append(trainX,traincategory,1)
testX = np.append(testX,testmissingprice,1)
testX = np.append(testX,testcategory,1)

print(trainX.shape)

# fit lasso model
with(parallel_backend('threading')):
    m = LassoCV(normalize=False, cv=5, verbose=True).fit(trainX, trainY)

# save model
file_name = 'linear_model.joblib'
dump(m, file_name)

def load_model():
    return load(file_name)

def plot():
    # show results for fit data
    plt.figure()
    ax = plt.subplot(111)
    plt.plot(m.alphas_, m.mse_path_, ':')
    plt.plot(m.alphas_, m.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
    plt.axvline(m.alpha_, linestyle='--', color='k', label='CV estimate')
    ax.set_xscale('log')
    plt.legend()
    plt.xlabel('$\lambda$')
    plt.ylabel('MSE')
    plt.axis('tight')
    plt.savefig('lasso_path.png')

    # show the terrible predictions
    testYpred = m.predict(testX)
    plt.figure()
    plt.plot(testY, testYpred, '.', alpha=0.1)
    plt.title('RMSE: %f' % np.sqrt(np.mean((testYpred - testY) ** 2)))
    plt.savefig('lasso_prediction.png')

plot()


