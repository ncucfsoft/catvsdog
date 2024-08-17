import torch.utils.data
import os   #文件操作模块
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

from torch.autograd import Variable
import torch.nn as nn

img_size = 200 #设置图片尺寸

dataset_dir = 'd:/python/validation/'  # 数据集路径
model_dir = 'd:/python/validation/model/'     # 网络参数保存位置
workers = 10                        # 线程数量
batch_size = 25                     # 一次训练所选取的样本数
lr = 0.001                         # 学习率
nepoch = 1                         # 训练的次数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#tran = Trans.Compose([Trans.Resize(img_size), Trans.CenterCrop([img_size, img_size]), Trans.ToTensor()]) #封装， 对后面读取的图片的格式转换

def train():
    datafile = DVCD('train', dataset_dir)   # 实例化数据集对象
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False)  # 创建数据加载器

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))
   
    model = Net().to(device)  # 实例化一个网络
    #model = nn.DataParallel(model)  # 多GPU并行化
    #modle=model.to('cuda')
    model.train()  # 将模型设置为训练模式
    print(torch.cuda.current_device())
    print(next(model.parameters()).is_cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam优化器
    Lossfuc = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    print(next(model.parameters()).is_cuda)
    cnt = 0  # 记录训练图片数量
    for epoch in range(nepoch):  # 遍历每个epoch
        #try:
         for img, label,filename in dataloader:  # 遍历每个batch
            
                img, label,filename = Variable(img), Variable(label),Variable(filename)  # 将数据转为PyTorch的Variable类型
           
                img=img.to(device)
                label=label.to(device)
                #print (str(label))
                out = model(img)  # 前向传播，计算输出
                #print ('model后:'+str(len(label)))
                #label=label.to(torch.float)
                newlabel=label
                loss=0
                if(len(label)>1):#batch_size必须设置大于1
                 newlabel=label.squeeze()
                 #print ('newlabel后:'+str(newlabel))
                 loss = Lossfuc(out,newlabel).to(device)  # 计算损失

                 loss.backward()  # 反向传播，计算梯度

                optimizer.step()  # 更新网络参数
                optimizer.zero_grad()  # 梯度清零
                cnt += 1

                print('Epoch:{0}, Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))
           
        #except:
          # print('fail img inload-----')   
           
    torch.save(model.state_dict(), '{0}/model.pth'.format(model_dir))  # 保存训练好的模型参数

if __name__ == "__main__":
    train()