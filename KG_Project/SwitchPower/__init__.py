import math
import config as conf

class SwitchPower():
    f_clk = 1e10
    i_clk = 1
    s = 1
    sin_theta = 0.7
    r = 0.1

    # -----------------每个IO通用属性------------
    @conf.auto_type_checker
    def __init__(self,f_clk: float, i_clk: float, s: float, sin_theta: float, r: float) :
        self.f_clk = f_clk
        self.i_clk = i_clk
        self.s = s
        self.sin_theta = sin_theta
        self.r = r


    def calcuRadiation(self):
        if(math.ceil(self.f_clk) & math.ceil(self.i_clk) & math.ceil(self.s) & math.ceil(self.sin_theta) & math.ceil(self.r)):
            return 'Radiation:'+str(131.6*10e-16*self.f_clk*self.i_clk*self.s*self.sin_theta/self.r)+'V/m'
        else:
            return None