
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import torch.nn as nn
class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.LR = KNeighborsRegressor()
    def train_model(self,x,y):
        self.LR.fit(x,y)
        return self.LR
    def test_model(self,x,y):
        self.score_r = self.LR.score(x, y)
        return self.score_r
    def forward(self,x):
        pred = self.LR.predict(x)
        return pred
