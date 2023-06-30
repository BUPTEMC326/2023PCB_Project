import math

import config as conf

class MicrostripLine():
    f_clk = 1e10
    i_clk = 1
    s = 1
    r = 0.1

    # -----------------每个IO通用属性------------
    @conf.auto_type_checker
    def __init__(self,f_clk: float, i_clk: float, s: float,  r: float) :
        self.f_clk = f_clk
        self.i_clk = i_clk
        self.s = s
        self.r = r


    def calcuDiffRadiation(self):
        if(math.ceil(self.f_clk) & math.ceil(self.i_clk) & math.ceil(self.s)  & math.ceil(self.r)):
            return 'Radiation:'+str(263*10e-16*self.f_clk*self.f_clk*self.i_clk*self.s/self.r)+'V/m'
        else:
            return None