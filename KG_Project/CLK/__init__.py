import math

import config as conf

class CLK():
    v0=1
    i0=1
    r=1

    # -----------------每个IO通用属性------------
    @conf.auto_type_checker
    def __init__(self,v0: float, i0: float, r: float) :
        self.v0 = v0
        self.i0 = i0
        self.r = r



    def calcuRadiation(self):
        if(math.ceil(self.v0) & math.ceil(self.i0) & math.ceil(self.r)):
            return 'Radiation:'+str((2.0/self.r)*math.sqrt(30*self.v0*self.i0))+'V/m'
        else:
            return None