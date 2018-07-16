from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import preprocessing
import numpy as np
from pandas import read_csv
import pandas as pd
from pandas import read_excel
from datetime import datetime
from matplotlib import pyplot
from numpy.random import randn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# def parse1(x):
# 	return datetime.strptime(x, '%Y')

# parse2 = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
# dataset1 = read_excel(r'E:\我的坚果云\屯溪流域\屯溪流量雨量资料\屯溪流域流量雨量.xlsx',sheetname=1,index_col=0,date_parser=parse2)
# dataset1.columns = ['liuliang', '1', '2', '3', '4', '5', '6', '7','8','9','10','11']
# dataset1.index.name = 'date'
# dataset1.to_csv('2001.csv')

dataset_train = read_csv('train.csv',header = 0,index_col=0)
# print(dataset_train.head(5))
values = dataset_train.values
where_value_nan = np.isnan(values)
values[where_value_nan] = 0
# groups=[0,1,2,3,4,5,6,7,8,9,10,11]
# i=1
# pyplot.figure()
# for group in groups:
# 	pyplot.subplot(len(groups),1,i)
# 	pyplot.plot(values[:,group])
# 	pyplot.title(dataset_train.columns[group],y = 0.5,loc = 'right')
# 	i+=1
# pyplot.show()

# print(values)
input_num ,output_num ,train_num= 8 , 6 ,10
def series_to_supervised(data, n_in, n_out, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	# print(n_vars)
	df = DataFrame(data)
	cols, names ,cols_out,names_out= list(), list(),list(),list()

	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		result_in = pd.concat(cols,axis=1)

		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		result_in.columns=names

	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols_out.append(df.shift(-i))

		# if i == 0:
		# 	names += [('var%d(t)' % (i+1))]
		# else:
		# 	names += [('var%d(t+%d)' % (i,n_out-1))]

		if i == 0:
			names_out += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names_out += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
		result_out = pd.concat(cols_out,axis=1)
		result_out.columns=names_out
	# put it all together
	agg = concat([result_in,result_out], axis=1)
	# agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
# load dataset


# integer encode direction
encoder = preprocessing.LabelEncoder()
# values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(values, input_num, output_num)
# drop columns we don't want to predict
# reframed.drop(reframed.columns[[13,14,15,16,17,18,19,20,21,22,23]], axis=1, inplace=True)

# print(reframed)

# ##画两条线
# df = reframed['var1(t)']
# df1 = reframed['var1(t+1)']
# result = pd.concat([df,df1],axis=1)
# result.plot()
# plt.show()

values = reframed.values
n_train_hours = 19000
#17500-18000（另一场洪水）
#19000-end （最后一场洪水）
train = values[:n_train_hours, :]
# print(train)
test = values[n_train_hours:, :]
# print(test)
# split into input and outputs
# scaler_train = preprocessing.MinMaxScaler(feature_range=(0, 1))
# scaler_train_y = preprocessing.MinMaxScaler(feature_range=(0, 1))
# scaler_test = preprocessing.MinMaxScaler(feature_range=(0, 1))
# scaler_test_y = preprocessing.MinMaxScaler(feature_range=(0, 1))

# train = scaler_train_x.fit_transform(train)
# test = scaler_train_x.fit_transform(test)



train_X_new = np.zeros((len(train),input_num*12+output_num))
test_X_new = np.zeros((len(test),input_num*12+output_num))
# 训练测试分开归一化
train_y = train[:, 12*(input_num+output_num-1)]
test_y = test[:, 12*(input_num+output_num-1)]

for i in range(0,12*input_num):
    train_X_new[:,i] = train[:,i]
    test_X_new[:,i] = test[:,i]


for j in range(0,output_num):
    train_X_new[:, j + 12*input_num] = np.average(train[:,12*(j+input_num)+1:12*(j+1+input_num)-1],axis=1)
    test_X_new[:, j + 12 * input_num] = np.average(test[:, 12 * (j + input_num) + 1:12 * (j + 1 + input_num) - 1],
                                                   axis=1)
train_y_new = train_y
test_y_new = test_y

train = concatenate((train_X_new,train_y_new.reshape(len(train_y),1)),axis=1)
test = concatenate((test_X_new,test_y_new.reshape(len(test_y),1)),axis=1)
train_maxmin = scaler.fit_transform(train)
test_maxmin = scaler.fit_transform(test)
train_X,train_y = train_maxmin[:,:-1],train_maxmin[:,-1]
test_X,test_y = test_maxmin[:,:-1],test_maxmin[:,-1]
# print(train)
# data_test = reframed.iloc[:,12*(input_num+output_num-1)]

# data_value = data_test.values
# train_v = scaler.inverse_transform(train_maxmin)

# train_X,train_y = train[:, 1:], train[:, 0]
# test_X, test_y = test[:, 1:], test[:, 0]


# test_y = scaler_test_y.fit_transform(test[:, 12*(input_num+output_num-1)])
# train_X, train_y = scaler.fit_transform(train[:, 0:input_num*4-1]), scaler.fit_transform(train[:, 12*(input_num+output_num-1)])
# test_X, test_y = scaler.fit_transform(test[:, 0:input_num*4-1]), scaler.fit_transform(test[:, 12*(input_num+output_num-1)])
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=train_num, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history

#打印损失函数
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((test_X,yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
# invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)


inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)

inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[: ,-1]
# max_peak = inv_y.index(max(inv_y))-inv_yhat.index(max(inv_yhat))
max_peak_time = inv_yhat.argmax(axis=0)-inv_y.argmax(axis=0)
max_peak_value = inv_yhat.max(axis = 0) - inv_y.max(axis = 0)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

# rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)
print('The max_peak_time: %d' % max_peak_time)
print('The max_peak_value: %d' % max_peak_value)
pyplot.figure()


pyplot.plot(inv_y,'b',label = 'obs')


pyplot.plot(inv_yhat,'g',label = 'pre')

pyplot.show()