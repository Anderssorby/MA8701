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
data['missing_price'] = data['price'].isna()
data['item_seq_number'].fillna(0, inplace=True)
data['description'].fillna(" ", inplace=True)
data['desc_length'] = data['description'].str.len()

# Split data into train and test data
train_data = data[:train_size]
test_data = data[train_size:]
data=0

train_price, train_missing_price, train_desc, train_desc_length, train_item_num, train_category, trainY = train_data['price'], train_data['missing_price'], train_data['description'], train_data['desc_length'], train_data['item_seq_number'], train_data['category_name'], train_data['deal_probability']

test_price, test_missing_price, test_desc, test_desc_length, test_item_num, test_category, testY = test_data['price'], test_data['missing_price'], test_data['description'], test_data['desc_length'], test_data['item_seq_number'], test_data['category_name'], test_data['deal_probability']
test_category = pd.get_dummies(pd.Categorical(test_category), prefix = 'category')

if price_ratio:
    df = pd.concat([train_price,train_category], axis=1)
    df.dropna(subset=['price'])
    category_means = np.asmatrix(df.groupby('category_name').mean()) #Calculate mean price within each category, NaNs excluded
    df = 0
else:
    train_price.fillna(0, inplace=True)
    test_price.fillna(0, inplace=True)

train_category = pd.get_dummies(pd.Categorical(train_category), prefix = 'category')

#Convert dataframes to numpy arrays
train_price = np.transpose(np.asmatrix(train_price))
test_price = np.transpose(np.asmatrix(test_price))
train_missing_price = np.transpose(np.asmatrix(train_missing_price))
test_missing_price = np.transpose(np.asmatrix(test_missing_price))
train_item_num = np.transpose(np.asmatrix(train_item_num))
test_item_num = np.transpose(np.asmatrix(test_item_num))
train_category = np.asmatrix(train_category)
test_category = np.asmatrix(test_category)
train_desc_length = np.transpose(np.asmatrix(train_desc_length))
test_desc_length = np.transpose(np.asmatrix(test_desc_length))

if price_ratio:
    for i in range(train_size): #Divide price by category mean from training set, NaN price ratios are set to 1
        cat = np.nonzero(train_category[i])[1][0]
        train_price[i] = 1 if math.isnan(train_price[i]) else train_price[i]/category_means[cat]

    for i in range(test_size): #Same for test set, using means from training set
        cat = np.nonzero(test_category[i])[1][0]
        test_price[i] = 1 if math.isnan(test_price[i]) else test_price[i]/category_means[cat]
else:
    train_pricecut = np.percentile(train_price,80) #Cuts prices larger than the 80th percentile
    for i in range(train_price.shape[0]): #Requires the test set to be at least as large as the training set
        if train_price[i,0] > train_pricecut:
            train_price[i,0] = train_pricecut
        if test_price[i,0] > train_pricecut:
            test_price[i,0] = train_pricecut

## Get "bag of words" transformation of the data
vec = TfidfVectorizer(ngram_range=(1, 1),
                      min_df=mdf,
                      max_df=0.9,
                      lowercase=True,
                      strip_accents='unicode',
                      sublinear_tf=True)

trainX = vec.fit_transform(train_desc)
testX = vec.transform(test_desc)

print(trainX.shape)

#plt.figure()
#plt.hist(train_price)
#plt.show()
#plt.plot(train_price, trainY, 'b.')
#plt.show()

#Add all numeric features
trainX = np.append(trainX.todense(),train_price,1)
trainX = np.append(trainX,train_item_num,1)
trainX = np.append(trainX,train_desc_length,1)
testX = np.append(testX.todense(),test_price,1)
testX = np.append(testX,test_item_num,1)
testX = np.append(testX,test_desc_length,1)

#Standardize numeric features
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

#Add categorical features
trainX = np.append(trainX,train_missing_price,1)
trainX = np.append(trainX,train_category,1)
testX = np.append(testX,test_missing_price,1)
testX = np.append(testX,test_category,1)

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


