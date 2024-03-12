# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:21:50 2022

@author: Maciej Rosoł
"""

#%% Results class

class ResultsNonlincausality():
    
    def __init__(self):
        self._models_X_all = []
        self._models_XY_all = []
        self._histories_X_all = []
        self._histories_XY_all = []
        self._RSS_X_all = []
        self._RSS_XY_all = []
        self._errors_X_all = []
        self._errors_XY_all = []
        
    @property
    def best_model_X(self):
        return self._best_model_X
    
    @best_model_X.setter
    def best_model_X(self, best_mdl_X):
        self._best_model_X = best_mdl_X
        
    @property
    def best_model_XY(self):
        return self._best_model_XY
    
    @best_model_XY.setter
    def best_model_XY(self, best_mdl_XY):
        self._best_model_XY = best_mdl_XY
        
    @property
    def models_X_all(self):
        return self._models_X_all
    
    def models_X_all_append(self, mdl_X):
        self._models_X_all.append(mdl_X)
        
    @property
    def models_XY_all(self):
        return self._models_XY_all
    
    def models_XY_all_append(self, mdl_X):
        self._models_XY_all.append(mdl_X)
        
    @property
    def best_history_X(self):
        return self._best_history_X
    
    @best_history_X.setter
    def best_history_X(self, best_history_X):
        self._best_history_X = best_history_X
        
    @property
    def best_history_XY(self):
        return self._best_history_XY
    
    @best_history_XY.setter
    def best_history_XY(self, best_history_XY):
        self._best_history_XY = best_history_XY
        
    @property
    def histories_X_all(self):
        return self._histories_X_all
    
    def histories_X_all_append(self, hist_X):
        self._histories_X_all.append(hist_X)
        
    @property
    def histories_XY_all(self):
        return self._histories_XY_all
    
    def histories_XY_all_append(self, hist_XY):
        self._histories_XY_all.append(hist_XY)
    
    @property
    def p_value(self):
        return self._p_value

    @p_value.setter
    def p_value(self, p_val):
        self._p_value = p_val
        
    @property
    def test_statistic(self):
        return self._test_statistic

    @test_statistic.setter
    def test_statistic(self, stat):
        self._test_statistic = stat
        
    @property
    def index_best_X(self):
        return self._index_best_X

    @index_best_X.setter
    def index_best_X(self, idx_X):
        self._index_best_X = idx_X
    
    @property
    def index_best_XY(self):
        return self._index_best_XY

    @index_best_XY.setter
    def index_best_XY(self, idx_XY):
        self._index_best_XY = idx_XY
        
    @property
    def RSS_X_all(self):
        return self._RSS_X_all
    
    @property
    def best_RSS_X(self):
        return self._best_RSS_X
    
    @best_RSS_X.setter
    def best_RSS_X(self, best_RRS_X):
        self._best_RSS_X = best_RRS_X
    
    def RSS_X_all_append(self, RSS_X):
        self._RSS_X_all.append(RSS_X)
        
    @property
    def RSS_XY_all(self):
        return self._RSS_XY_all
    
    def RSS_XY_all_append(self, RSS_XY):
        self._RSS_XY_all.append(RSS_XY)
        
    @property
    def best_RSS_XY(self):
        return self._best_RSS_XY
    
    @best_RSS_XY.setter
    def best_RSS_XY(self, best_RRS_XY):
        self._best_RSS_XY = best_RRS_XY
        
    @property
    def errors_X_all(self):
        return self._errors_X_all
    
    def errors_X_all_append(self, error_X):
        self._errors_X_all.append(error_X)
        
    @property
    def errors_XY_all(self):
        return self._errors_XY_all
    
    def errors_XY_all_append(self, error_XY):
        self._errors_XY_all.append(error_XY)
        
    @property
    def best_errors_X(self):
        return self._best_errors_X
    
    @best_errors_X.setter
    def best_errors_X(self, best_errors_X):
        self._best_errors_X = best_errors_X
        
    @property
    def best_errors_XY(self):
        return self._best_errors_XY
    
    @best_errors_XY.setter
    def best_errors_XY(self, best_errors_XY):
        self._best_errors_XY = best_errors_XY
    
    def append_results(self, RSS_X, RSS_XY, model_X, model_XY, history_X, history_XY, error_X, error_XY):
        self.RSS_X_all_append(RSS_X)
        self.RSS_XY_all_append(RSS_XY)
        self.models_X_all_append(model_X)
        self.models_XY_all_append(model_XY)
        self.histories_X_all_append(history_X)
        self.histories_XY_all_append(history_XY)
        self.errors_X_all_append(error_X)
        self.errors_XY_all_append(error_XY)
        
    def set_best_results(self, idx_bestX, idx_bestXY):
        self.index_best_X = idx_bestX 
        self.index_best_XY = idx_bestXY
        self.best_model_X = self.models_X_all[idx_bestX]
        self.best_model_XY = self.models_XY_all[idx_bestXY]
        self.best_history_X = self.histories_X_all[idx_bestX]
        self.best_history_XY = self.histories_XY_all[idx_bestXY]
        self.best_RSS_X = self.RSS_X_all[idx_bestX]
        self.best_RSS_XY = self.RSS_XY_all[idx_bestXY]
        self.best_errors_X = self.errors_X_all[idx_bestX]
        self.best_errors_XY = self.errors_XY_all[idx_bestXY]
        
