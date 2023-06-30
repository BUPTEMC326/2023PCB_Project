import config as conf


class IC():
    NCpin={'tvs':0}


    # -----------------每个IO通用属性------------
    @conf.auto_type_checker
    def __init__(self,NCpin:dict):
        self.NCpin = NCpin

    def NCPIN(self):
        if self.NCpin['tvs']==0:
            return 'NC pin no TVS'
        else:
            return 'NC pin has TVS'

class NCpin():
    # -----------------每个IO通用属性------------
    @conf.auto_type_checker
    def __init__(self, NCpin: dict):
        self.NCpin = NCpin