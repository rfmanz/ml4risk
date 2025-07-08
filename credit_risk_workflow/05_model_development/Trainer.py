import os, sys
import numpy as np
import pandas as pd
import pickle as pkl
from abc import ABCMeta, abstractmethod
from pathlib import Path
from smart_open import open    

class TrainerBase(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        for k in kwargs:
            setattr(k, kwargs[k])
            
    @abstractmethod
    def train(self, model, train_data, valid_data=None, **kwargs):
        raise NotImplementedError
    
    
    @abstractmethod
    def validate(self, valid_data):
        raise NotImplementedError
    
    
    @abstractmethod
    def predict(self, test_data):
        raise NotImplementedError
        
    
    @abstractmethod
    def save_model(self, path):
        raise NotImplementedError
        

class LGBMTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        self.model = None
        self.features = None
        self.target = None
        super().__init__(*args, **kwargs)
        
        
    def train(self, model, train_df, features, target_col, valid_df=None, **kwargs):
        import lightgbm as lgb
        
        if isinstance(model, lgb.LGBMClassifier):
            eval_set = None
            if valid_df is not None:
                eval_set = (valid_df[features], valid_df[target_col])
            model.fit(train_df[features],
                      train_df[target_col],
                      eval_set=eval_set,
                      **kwargs)
            self.model = model
            self.features = features
            self.target = self.target
        else:
            raise NotImplementedError("Currently only support lgb.LGBMClassifier")
    
    def validate(self, valid_df, features=None):
        import lightgbm as lgb
        raise NotImplementedError
            
    
    def predict(self, test_df, features=None):
        import lightgbm as lgb
        if isinstance(self.model, lgb.LGBMClassifier):
            features = features if features is not None else self.features
            return self.model.predict_proba(test_df[features])
        else:
            raise NotImplementedError
            
    
    def save_model(self, path):
        p = Path(path)
        p.parents[0].mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pkl.dump(self.model, f)
            
            
# TODO: Keras Tabular MLP/TABNET/NEURAL ADDITIVE TRAINER
class TFTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        self.model = None
        self.history = None
        super().__init__(*args, **kwargs)
        
    def train(self, model, train_data,
              epochs, validation_data=None,
              callbacks=[], 
              **kwargs):
        
        # fit model 
        history = model.fit(train_data,
                              validation_data=validation_data,
                              epochs=epochs,
                              callbacks=callbacks,
                              **kwargs)
        
        self.model = model
        self.history = history
        return history
        
    def validate(self, valid_data, return_dict=True, **kwargs):
        import lightgbm as lgb
        
        return self.model.evaluate(valid_data, return_dict=return_dict, **kwargs)
    
    def predict(self, test_data):
        import lightgbm as lgb
        
        return self.model.predict(test_data)
    
    def save_model(self, path):
        p = Path(path)
        p.parents[0].mkdir(parents=True, exist_ok=True)
        self.model.save(path)
            
    @staticmethod
    def df_to_tensor(x, y=None, batch_size=256, shuffle=True,
                     buffer_size=5000, repeat=False,
                     dtype=np.float32):
        import pandas as pd
        import tensorflow as tf

        data = (x,y) if y is not None else x
        data = tf.data.Dataset.from_tensor_slices(data)
        if repeat:
            data = data.repeat()
        if shuffle:  # don't shuffle for train/test
            data = data.shuffle(buffer_size)  
        data = (data.batch(batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE))
        return data
        
    
