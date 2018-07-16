import scipy.io as scio
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
data_2010 = scio.loadmat('data_2010_7_15to7_30.mat')
data_2011 = scio.loadmat('data_2011_7_2to9_30.mat')
data_2012 = scio.loadmat('data_2012_7_1to9_20.mat')
data_2013 = scio.loadmat('data_2013_5_5to9_30.mat')
data_2014 = scio.loadmat('data_2014_8_5to9_30.mat')
# values = concatenate((data_2010,data_2011),axis=1)


input_num ,output_num ,train_num= 8 , 6 ,10
def series_to_supervised(data, n_in, n_out, dropnan=True):
	# n_vars = 1 if type(data) is list else data.shape[1]
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

reframed = series_to_supervised(data_2010, input_num, output_num)
print(data)