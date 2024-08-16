import torch.utils.data
import os   #文件操作模块
import torch
import torch.utils.data as data   #用于继承一个父类（data.Dataset）里的函数
from PIL import Image
import torchvision.transforms as Trans #在定义图片转换的格式时，会用到相关的函数

img_size = 200 #设置图片尺寸

tran = Trans.Compose([Trans.Resize(img_size), Trans.CenterCrop([img_size, img_size]), Trans.ToTensor()]) #封装， 对后面读取的图片的格式转换

class DogsVSCatsDataset(data.Dataset):
    def __init__(self, mode, dir):
        self.data_size = 0 #数据集的大小
        self.img_list=[] #用于存图
        self.img_label =[]#标签
        self.img_filename=[]#文件名
        self.trans=tran #转换的属性设置
        self.mode =mode #下面打开集的模式

        if self.mode =='train' or self.mode =='val':
            catdir=dir+'train/cat/'
            dogdir=dir+'train/dog/'
            for file in os.listdir(catdir): #遍历
                 if not os.path.isdir(dir+file):
                    if file.lower().find(".db") == -1:
                        self.img_list.append(catdir+file) #存图
                        self.data_size += 1
                        name = file.split(sep='.')[0] #将 该文件名拆分，便于判断是cat还是dog
                        filenameid=name.split(sep='/')[-1]
                        label_x =0
                        #print(file)               
                        
                        self.img_label.append(label_x)#设置入相对于的标签；cat:1； dog:0
                        finint=0
                        try:
                            finint=label_x*100000+int(filenameid)
                        except:
                            finint=0   
                        self.img_filename.append(finint)
            for file in os.listdir(dogdir): #遍历
                 if not os.path.isdir(dir+file):
                    if file.lower().find(".db") == -1:
                        self.img_list.append(dogdir+file) #存图
                        self.data_size += 1
                        name = file.split(sep='.')[0] #将 该文件名拆分，便于判断是cat还是dog
                        filenameid=name.split(sep='/')[-1]
                        label_x =1
                        #print(file)               
                        self.img_label.append(label_x)#设置入相对于的标签；cat:1； dog:0
                        finint=0
                        try:
                            finint=label_x*100000+int(filenameid)
                        except:
                            finint=0   
                        self.img_filename.append(finint)

        elif self.mode == 'test':
            dir +='test/'
            for file in os.listdir(dir):#同理
                if not os.path.isdir(dir+file):
                    self.img_list.append(dir+file)
                    name = file.split(sep='.')[0] #将 该文件名拆分，便于判断是cat还是dog
                    filenameid=name.split(sep='/')[-1]
                    self.data_size +=1
                    self.img_label.append(2)#无意义的标签
                    finint=0
                    try:
                     finint=label_x*100000+int(filenameid)
                    except:
                     finint=0   
    
                    self.img_filename.append(finint)
        else:
            print("没有这个mode")
        #print("init over")    

    def __getitem__(self,item):  #获取数据
        if self.mode =='train' or self.mode=='test' or self.mode=='val':
            #print(f'train begin{self.img_list[item]}')
            filename=self.img_filename[item]
            try:
             img =Image.open(self.img_list[item]).convert('RGB')
             
             if self.mode =='train' or self.mode =='val':
              label_y = self.img_label[item]
              return self.trans(img), torch.LongTensor([label_y]), torch.Tensor([filename]) #返回该图片的地址和标签
             elif self.mode=='test':
              img =Image.open(self.img_list[item]).convert('RGB')
              return self.trans(img)
            except Image.UnidentifiedImageError:
             print(f"无法识别的图像文件：{self.img_list[item]}")
             return None
           
        else:
            print("None")
            return None
    def __len__(self):
        return self.data_size
