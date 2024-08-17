import torch.utils.data
import shutil
import torch
import torch.utils.data as data   #用于继承一个父类（data.Dataset）里的函数
from PIL import Image
import torchvision.transforms as Trans #在定义图片转换的格式时，会用到相关的函数
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from getdata import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from network import Net
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

img_size = 200 #设置图片尺寸
bCopyerrorimg=False

model_file = 'd:/python/validation/model/model.pth'  # 模型文件路径
dataset_dir = 'd:/python/validation/'  # 数据集路径
model_dir = 'd:/python/validation/model/'     # 网络参数保存位置
workers = 10                        # 线程数量
batch_size = 10                     # 一次训练所选取的样本数
lr = 0.001                         # 学习率
nepoch = 20                         # 训练的次数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#tran = Trans.Compose([Trans.Resize(img_size), Trans.CenterCrop([img_size, img_size]), Trans.ToTensor()]) #封装， 对后面读取的图片的格式转换

def val():

    model = Net().to(device)  # 实例化一个网络模型
    #model = nn.DataParallel(model)  # 多GPU并行化
    model = model.to(device)
    model.load_state_dict(torch.load(model_file,weights_only=True))  # 加载模型参数
    model.eval()  # 将模型设置为评估模式
    datafile = DVCD('val', dataset_dir)  # 实例化测试数据集对象
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False)  # 创建数据加载器

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))
   
    model.train(False)
    print(torch.cuda.current_device())
    print(next(model.parameters()).is_cuda)
    total,correct = 1,1
    #torch.no_grad()
    print('datasizeof: '+str(dataloader.__sizeof__()))
    #try:
    for img, label,filename in dataloader:
                img, label,filename = Variable(img), Variable(label),Variable(filename)  # 将数据转为PyTorch的Variable类型
                    
                img=img.to(device)
                #print('img:'+str(img))
                label=label.to(device)
                filename=filename.to(device)
               # print('label:'+str(label))
                out = model(img)
                noum,predicted = torch.max(out.data,dim=1)
                #print('out: '+str(out.data))
                #print('noum: '+str(noum.data))
                #print('predict: '+str(predicted.data))
                label=label.reshape(-1)
                total += label.size(0)
                #print("predstr:"+str(predicted))
                #print("labelstr:"+str(label))
                      
                curcorrect=(predicted==label).sum().item() 
                #if(label.size(0)>1):
                # print('labelsize------:%d curcorrect:%d predictsize:%d'%(label.size(0),curcorrect,predicted.size(0)))
                
                correct += curcorrect
                if (bCopyerrorimg):
                    for i in range(predicted.size(0)):
                    # print('labeli:%d filenamei %d'%(int(label[i]),int(filename[i])))
                         if(predicted[i]!=label[i]):
                          if(int(filename[i])<100000):
                              caterror=dataset_dir+"error/cat/"+str(int(filename[i]))+".jpg"
                              catorgfpath=dataset_dir+"train/cat/"+str(int(filename[i]))+".jpg"
                              #print (caterror)
                              shutil.copyfile(catorgfpath,caterror)
                          else:
                              caterror=dataset_dir+"error/dog/"+str(int(filename[i])-100000)+".jpg"
                              catorgfpath=dataset_dir+"train/dog/"+str(int(filename[i])-100000)+".jpg"
                              #print (caterror)
                              shutil.copyfile(catorgfpath,caterror)
                          print('not equal------------------'+str(int(filename[i])))
    #except:
    # print('fail img inload-----')   
    print('正确率: %d %% %d %d '% (100*correct/total,correct,total))
if __name__ == "__main__":
    val()