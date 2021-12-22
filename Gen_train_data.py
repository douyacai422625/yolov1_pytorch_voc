import os

class Gen_VOC_data:
    def __init(self):
        self.a = 1
    def data_process(self,path):
        img_list = os.listdir(os.path.join(path,'JPEGImages'))
        
