# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 21:12:00 2022

@author: Maciej Rosoł
"""
import pytest

import numpy as np
import keras
import nonlincausality_new03022022 as nlc
#%% Test check input

class TestCheckInput:
    
    test_data_ok = [
        (np.zeros([1000,2]), [5], 2, [10,10], 2, [10,10], [], 1, [], [], True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        (np.zeros([1000,2]), [5], 2, [10,10], 2, [10,10], [], 1, [], [], True, 0.1, [10,10], [0.01,0.001], 8, False, False, 'LSTM'),
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], np.zeros([100,2]), 1, [], [], True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], np.zeros([100,2]), 1, np.ones([1000,5]), np.ones([100,2]), True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], np.zeros([100,2]), 1, [], [], True, 0.1, 10, 0.01, 8, True, False, 'GRU'),
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], np.zeros([100,2]), 1, [], [], True, 0.1, 10, 0.01, 8, False, True, 'GRU'),
        (np.zeros([1000,2]), 5, 1, [10], 2, [10,10], np.zeros([100,2]), 1, [], [], True, 0.1, 10, 0.01, 8, False, False, 'GRU'),
        (np.zeros([1000,2]), [5], ['d','dr','l','dr','g','dr'], [10, 0.1, 10, 0.1, 10, 0.1], None, None, np.zeros([100,2]), 1, [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        (np.zeros([1000,2]), [5], None, None, 2, [10,10], np.zeros([100,2]), 1, [], [], True, 0.1, 10, 0.01, 8, False, False, 'MLP'),
        ]
    
    @pytest.mark.parametrize('x, maxlag, Network_layers, Network_neurons, Dense_layers, Dense_neurons, xtest, run, z, ztest, add_Dropout, Dropout_rate, epochs_num, learning_rate, batch_size_num, verbose, plot, functin_type',test_data_ok)
    def test_check_input_ok(self,x, maxlag, Network_layers, Network_neurons, Dense_layers, Dense_neurons, xtest, run, z, ztest, add_Dropout, Dropout_rate, epochs_num, learning_rate, batch_size_num, verbose, plot, functin_type):
        print(nlc.check_input(x, maxlag, Network_layers, Network_neurons, Dense_layers, Dense_neurons, xtest, run, z, ztest, add_Dropout, Dropout_rate, epochs_num, learning_rate, batch_size_num, verbose, plot, functin_type))
    
    test_data_not_ok = [
        # to many columns in x
        (np.zeros([1000,3]), [5], 2, [10,10], 2, [10,10], 1, [], [], [], True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        # epoch_num is list and learning_rate is int
        (np.zeros([1000,2]), [5], 2, [10,10], 2, [10,10], 1, [], [], [], True, 0.1, [10,10], 0.01, 8, False, False, 'LSTM'),
        # epoch_num and learning_rate have different length 
        (np.zeros([1000,2]), [5], 2, [10,10], 2, [10,10], 1, [], [], [], True, 0.1, [10,10], [0.01,0.001,0.001], 8, False, False, 'LSTM'),
        # negative epochs_num
        (np.zeros([1000,2]), [5], 2, [10,10], 2, [10,10], 1, [], [], [], True, 0.1, [-10,10], [0.01,0.001], 8, False, False, 'LSTM'),
        # negative learning_rate
        (np.zeros([1000,2]), [5], 2, [10,10], 2, [10,10], 1, [], [], [], True, 0.1, [10,10], [-0.01,0.001], 8, False, False, 'LSTM'),
        # run = 0
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], 0, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        # run <0
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], -1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        # z is to short
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], 1, np.zeros([100,2]), np.ones([5000,5]), np.ones([100,2]), True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        # ztest is to short
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], 1, np.zeros([100,2]), np.ones([1000,5]), np.ones([50,2]), True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        # batch_size = 0
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 0, True, False, 'LSTM'),
        # batch_size <0
        (np.zeros([1000,2]), [4,5], 1, [10], 2, [10,10], 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, -8, True, False, 'LSTM'),
        # maxlag = 0
        (np.zeros([1000,2]), 0, 1, [10], 2, [10,10], 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        # maxlag < 0
        (np.zeros([1000,2]), -5, 1, [10], 2, [10,10], 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'LSTM'),
        # incorrect functin_type
        (np.zeros([1000,2]), 5, 1, [10], 2, [10,10], 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'AAA'),
        # incorrect NN_config
        (np.zeros([1000,2]), 5, ['a','1'], [10,10], None, None, 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        # incorrect NN_config Dense=0.1
        (np.zeros([1000,2]), 5, ['d'], [0.1], None, None, 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        # incorrect NN_config dr>1
        (np.zeros([1000,2]), 5, ['dr'], [1.1], None, None, 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        # incorrect NN_config dr<0
        (np.zeros([1000,2]), 5, ['dr'], [-0.1], None, None, 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        # incorrect NN_config
        (np.zeros([1000,2]), 5, ['g'], [0.1], None, None, 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        # incorrect NN_config
        (np.zeros([1000,2]), 5, ['l'], [0.1], None, None, 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        # incorrect NN_config
        (np.zeros([1000,2]), 5, ['g'], [-0.1], None, None, 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        # incorrect NN_config
        (np.zeros([1000,2]), 5, ['l'], [-0.1], None, None, 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        # incorrect NN_config
        (np.zeros([1000,2]), 5, None, None, None, None, 1, np.zeros([100,2]), [], [], True, 0.1, 10, 0.01, 8, False, False, 'NN'),
        ]
    @pytest.mark.parametrize('x, maxlag, Network_layers, Network_neurons, run, Dense_layers, Dense_neurons, xtest, z, ztest, add_Dropout, Dropout_rate, epochs_num, learning_rate, batch_size_num, verbose, plot, functin_type',test_data_not_ok)
    def test_check_input_not_ok(self,x, maxlag, Network_layers, Network_neurons, Dense_layers, Dense_neurons, xtest, run, z, ztest, add_Dropout, Dropout_rate, epochs_num, learning_rate, batch_size_num, verbose, plot, functin_type):
        error = False
        try:
            nlc.check_input(x, maxlag, Network_layers, Network_neurons, Dense_layers, Dense_neurons, xtest, run, z, ztest, add_Dropout, Dropout_rate, epochs_num, learning_rate, batch_size_num, verbose, plot, functin_type)
        except:
            error = True
        assert error

#%% test MLP_architecture

class TestMLPArchitecture():
    test_data = [
        (2, [10,10], True, 0.1, [100, 5, 1], 7, 181),
        (2, [10,10], True, 0.1, [100, 5, 2], 7, 191),
        (2, [10,10], False, 0.1, [100, 5, 1], 5, 181),
        (2, [10,10], False, 0.1, [100, 5, 2], 5, 191),
        (1, [10], True, 0.1, [100, 5, 1], 5, 71),
        (1, [10], False, 0.1, [100, 5, 1], 4, 71),
        (1, [10], True, 0.1, [100, 5, 2], 5, 81),
        (1, [10], False, 0.1, [100, 5, 2], 4, 81),
        (3, [10,10,10], True, 0.1, [100, 5, 1], 9, 291),
        (3, [10,10,10], False, 0.1, [100, 5, 1], 6, 291),
        (3, [10,10,10], True, 0.1, [100, 5, 2], 9, 301),
        (3, [10,10,10], False, 0.1, [100, 5, 2], 6, 301),
        ]
    
    @pytest.mark.parametrize('Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape, expected_layers_number, expected_parameters', test_data)
    def test_MLP_architecture(self, Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape, expected_layers_number, expected_parameters):
        mdl = nlc.MLP_architecture(Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape)
        assert len(mdl.layers) == expected_layers_number
        assert mdl.count_params() == expected_parameters
        
#%% test GRU_architecture
class TestGRUArchitecture():
    test_data = [
        (2, [10,10], 1, [10], True, 0.1, [100, 5, 1], 8, 1111),
        (2, [10,10], 2, [10,10], True, 0.1, [100, 5, 2], 10, 1251),
        (2, [10,10], 3, [10,10, 10], False, 0.1, [100, 5, 1], 7, 1331),
        (2, [10,10], 0, [], False, 0.1, [100, 5, 2], 4, 1031),
        (1, [10], 1, [10], True, 0.1, [100, 5, 1], 5, 481),
        (1, [10], 2, [10,10], False, 0.1, [100, 5, 1], 5, 591),
        (1, [10], 3, [10,10, 10], True, 0.1, [100, 5, 2], 9, 731),
        (1, [10], 0, [], False, 0.1, [100, 5, 2], 3, 401),
        (3, [10,10,10], 1, [10], True, 0.1, [100, 5, 1], 10, 1741),
        (3, [10,10,10], 2, [10,10], False, 0.1, [100, 5, 1], 7, 1851),
        (3, [10,10,10], 3, [10,10, 10], True, 0.1, [100, 5, 2], 14, 1991),
        (3, [10,10,10], 0, [], False, 0.1, [100, 5, 2], 5, 1661),
        ]
    
    @pytest.mark.parametrize('GRU_layers, GRU_neurons, Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape, expected_layers_number, expected_parameters', test_data)
    def test_GRU_architecture(self, GRU_layers, GRU_neurons, Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape, expected_layers_number, expected_parameters):
        mdl = nlc.GRU_architecture(GRU_layers, GRU_neurons, Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape)
        assert len(mdl.layers) == expected_layers_number
        assert mdl.count_params() == expected_parameters

#%%
class TestLSTMArchitecture():
    test_data = [
        (2, [10,10], 1, [10], True, 0.1, [100, 5, 1], 8, 1441),
        (2, [10,10], 2, [10,10], True, 0.1, [100, 5, 2], 10, 1591),
        (2, [10,10], 3, [10,10, 10], False, 0.1, [100, 5, 1], 7, 1661),
        (2, [10,10], 0, [], False, 0.1, [100, 5, 2], 4, 1371),
        (1, [10], 1, [10], True, 0.1, [100, 5, 1], 5, 601),
        (1, [10], 2, [10,10], False, 0.1, [100, 5, 1], 5, 711),
        (1, [10], 3, [10,10, 10], True, 0.1, [100, 5, 2], 9, 861),
        (1, [10], 0, [], False, 0.1, [100, 5, 2], 3, 531),
        (3, [10,10,10], 1, [10], True, 0.1, [100, 5, 1], 10, 2281),
        (3, [10,10,10], 2, [10,10], False, 0.1, [100, 5, 1], 7, 2391),
        (3, [10,10,10], 3, [10,10, 10], True, 0.1, [100, 5, 2], 14, 2541),
        (3, [10,10,10], 0, [], False, 0.1, [100, 5, 2], 5, 2211),
        ]
    
    @pytest.mark.parametrize('LSTM_layers, LSTM_neurons, Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape, expected_layers_number, expected_parameters', test_data)
    def test_LSTM_arcgitecture(self, LSTM_layers, LSTM_neurons, Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape, expected_layers_number, expected_parameters):
        mdl = mdl = nlc.LSTM_architecture(LSTM_layers, LSTM_neurons, Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape)
        assert expected_layers_number == len(mdl.layers)
        assert expected_parameters == mdl.count_params()
        
'''
for i in range(len(test_data)):
    LSTM_layers, LSTM_neurons, Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape, exp_l, exp_par = test_data[i]
    mdl = nlc.LSTM_architecture(LSTM_layers, LSTM_neurons, Dense_layers, Dense_neurons, add_Dropout, Dropout_rate, data_shape)
    print(exp_l == len(mdl.layers))
    print(exp_par == mdl.count_params())
'''
#%% test NN architecture

class TestNNArchitecture():
    test_data = [
        (['d'], [10], [100, 5, 1], 4, 71),
        (['d','d'], [10, 10], [100, 5, 1], 5, 181),
        (['d'], [10], [100, 5, 2], 4, 81),
        (['d','d'], [10, 10], [100, 5, 2], 5, 191),
        (['d','dr'], [10,0.1], [100, 5, 1], 5, 71),
        (['d','dr','d','dr'], [10, 0.1, 10, 0.1], [100, 5, 1], 7, 181),
        (['d', 'dr'], [10, 0.1], [100, 5, 2], 5, 81),
        (['d','dr','d','dr'], [10, 0.1, 10, 0.1], [100, 5, 2], 7, 191),
        (['g'], [10], [100, 5, 1], 3, 371),
        (['g','d'], [10, 10], [100, 5, 1], 4, 481),
        (['g','d','g'], [10, 10, 10], [100, 5, 1], 5, 1111),
        (['d','g','d'], [10, 10, 10], [100, 5, 1], 5, 771),
        (['d','dr','g','dr','d','dr'], [10, 0.1, 10, 0.1, 10, 0.1], [100, 5, 1], 8, 771),
        (['g','d','g'], [10, 10, 10], [100, 5, 2], 5, 1141),
        (['d','g','d'], [10, 10, 10], [100, 5, 2], 5, 781),
        (['d','dr','g','dr','d','dr'], [10, 0.1, 10, 0.1, 10, 0.1], [100, 5, 2], 8, 781),
        (['l'], [10], [100, 5, 1], 3, 491),
        (['l','d'], [10, 10], [100, 5, 1], 4, 601),
        (['l','d','l'], [10, 10, 10], [100, 5, 1], 5, 1441),
        (['d','l','d'], [10, 10, 10], [100, 5, 1], 5, 981),
        (['d','dr','l','dr','d','dr'], [10, 0.1, 10, 0.1, 10, 0.1], [100, 5, 1], 8, 981),
        (['l','d','l'], [10, 10, 10], [100, 5, 2], 5, 1481),
        (['d','l','d'], [10, 10, 10], [100, 5, 2], 5, 991),
        (['d','dr','l','dr','d','dr'], [10, 0.1, 10, 0.1, 10, 0.1], [100, 5, 2], 8, 991),
        (['g','dr','l','dr','d','dr'], [10, 0.1, 10, 0.1, 10, 0.1], [100, 5, 1], 8, 1321),
        (['d','dr','l','dr','g','dr'], [10, 0.1, 10, 0.1, 10, 0.1], [100, 5, 1], 8, 1501),
        (['d','dr','l','dr','g','dr','d'], [10, 0.1, 10, 0.1, 10, 0.1, 10], [100, 5, 1], 9, 1611),
        (['g','dr','l','dr','d','dr'], [10, 0.1, 10, 0.1, 10, 0.1], [100, 5, 2], 8, 1351),
        (['d','dr','l','dr','g','dr'], [10, 0.1, 10, 0.1, 10, 0.1], [100, 5, 2], 8, 1511),
        (['d','dr','l','dr','g','dr','d'], [10, 0.1, 10, 0.1, 10, 0.1, 10], [100, 5, 2], 9, 1621),    
        ]
    
    @pytest.mark.parametrize('NN_config, NN_neurons, data_shape, expected_layers_number, expected_parameters', test_data)
    def test_NN_arcgitecture(self, NN_config, NN_neurons, data_shape, expected_layers_number, expected_parameters):
        mdl = mdl = nlc.NN_architecture(NN_config, NN_neurons, data_shape)
        assert expected_layers_number == len(mdl.layers)
        assert expected_parameters == mdl.count_params()


#%%


if __name__ == "__main__":
    pytest.main([__file__])
    
