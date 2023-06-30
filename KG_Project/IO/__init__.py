class IO():
    EAF=[]
    ESDprot = 0
    filter = 0


    # -----------------每个IO通用属性------------
    def __init__(self,ESDprot,filter):
        self.EAF=['有ESD保护器件和滤波器','有ESD保护器件无滤波器','无ESD保护器件有滤波器','无ESD保护器件无滤波器']
        self.ESDprot = ESDprot
        self.filter = filter




    def f_ESDAndFilter(self):
        ans=''
        if self.ESDprot>0:
            if self.filter>0:
                ans+=self.EAF[0]
            else:
                ans+=self.EAF[1]
        else:
            if self.filter>0:
                ans+=self.EAF[2]
            else:
                ans+=self.EAF[3]
        return ans


