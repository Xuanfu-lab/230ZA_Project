import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K


class Model_LSTM:
    def __init__(self, loss = "paper", reg = False):
        self.data = None
        self.model = None
        self.loss = loss
        self.reg = reg
        

    def __build_model(self, input_shape, outputs):
        '''
        Builds and returns the Deep Neural Network that will compute the allocation ratios
        that optimize the Sharpe Ratio of the portfolio
        
        inputs: input_shape - tuple of the input shape, outputs - the number of assets
        returns: a Deep Neural Network model
        '''
        if self.reg == False:
            model = Sequential([
                LSTM(64, input_shape=input_shape),
                Flatten(),
                Dense(outputs, activation='softmax')
            ])
        else:
            model = Sequential([
                LSTM(64, input_shape=input_shape),
                Flatten(),
                Dense(outputs, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(10))
            ])
            
            
        def sharpe_loss(_, y_pred):
            # make all time-series start at 1
            data = tf.divide(self.data, self.data[0])  
            
            # value of the portfolio after allocations applied
            portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1) 
            
            portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]  # % change formula
            
            # mean = tf.reduce_mean(portfolio_returns, axis=0)
            # stddev = tf.math.reduce_std(portfolio_returns, axis=0)
            # portfolio_returns = (portfolio_returns - mean) / stddev
            
            if self.loss == "paper":
                loss = K.mean(portfolio_returns) / K.std(portfolio_returns)
            elif self.loss == "return":
                loss = K.mean(portfolio_returns)
            elif self.loss == "convex":
                loss = K.mean(portfolio_returns) - K.std(portfolio_returns)

            # since we want to maximize Sharpe, while gradient descent minimizes the loss, 
            #   we can negate Sharpe (the min of a negated function is its max)
            return -loss
        
        model.compile(loss=sharpe_loss, optimizer='adam')
        return model
    
    
    def get_allocations(self, data: pd.DataFrame):
        '''
        Computes and returns the allocation ratios that optimize the Sharpe over the given data
        
        input: data - DataFrame of historical closing prices of various assets
        
        return: the allocations ratios for each of the given assets
        '''
        
        # data with returns
        data_w_ret = np.concatenate([ data.values[1:], data.pct_change().values[1:] ], axis=1)
        
        data = data.iloc[1:]
        self.data = tf.cast(tf.constant(data), float)
        
        if self.model is None:
            self.model = self.__build_model(data_w_ret.shape, len(data.columns))
        
        fit_predict_data = data_w_ret[np.newaxis,:]        
        self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=20, shuffle=False,
                       verbose=0
                      )
        return self.model.predict(fit_predict_data)[0]
    
    
