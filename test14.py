from getdata import DogsVSCatsDataset as DVCD
from network import Net
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shutil
from torch.utils.data import DataLoader as DataLoader
from PIL import Image
import os
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止打印图片出错
dataset_dir = 'd:/python/validation/'  # 测试集路径
model_file = 'd:/python/validation/model/model.pth'  # 模型文件路径
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
workers = 10                        # 线程数量
batch_size = 10                     # 一次训练所选取的样本数


def main():
    #N = 10  # 随机选择测试图片的数量
    model = Net()  # 实例化一个网络模型
    #model = nn.DataParallel(model)  # 多GPU并行化
    model = model.to(device)
    model.load_state_dict(torch.load(model_file,weights_only=True))  # 加载模型参数
    model.eval()  # 将模型设置为评估模式
    datafile = DVCD('test', dataset_dir)  # 实例化测试数据集对象
    #dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False)  # 创建数据加载器

    #index = np.random.randint(0, datafile.data_size, 1)[0]  # 随机选择一个索引
    #img = datafile.__getitem__(index).to(device)  # 获取对应索引的图片及其标签
    
    index = np.random.randint(0, datafile.data_size, 1)[0]  # 随机选择一个索引
    img = datafile.__getitem__(index).to(device)
    img = img.unsqueeze(0)  # 在第0维增加一个维度，变成(batch_size, channels, height, width)
    img = Variable(img)  # 将数据转为PyTorch的Variable类型
    out = model(img).to(device)  # 输入模型，进行前向传播，得到输出
    out = F.softmax(out, dim=1)  # 对输出进行softmax处理，得到分类概率
     
    if(os.path.exists(dataset_dir+"test/cat/")):
        for n in range(datafile.data_size):    
            print ("n:%d file:%s"%(n,datafile.img_list[n]))
            batchimg = datafile.__getitem__(n).to(device)  # 获取对应索引的图片及其标签
            batchimg = batchimg.unsqueeze(0)  # 在第0维增加一个维度，变成(batch_size, channels, height, width)
            batchimg = Variable(batchimg)  # 将数据转为PyTorch的Variable类型
            batchout = model(batchimg).to(device)  # 输入模型，进行前向传播，得到输出
            _,predicted = torch.max(batchout.data,dim=1)
            
            #out = F.softmax(out, dim=1)  # 对输出进行softmax处理，得到分类概率
            #print(out.data)
            #print("out1:"+str(out[0, 0])+"out2:"+str(out[0,1]))
            curfilename=datafile.img_list[n].split(sep='/')[-1]
            if(int(predicted[0])==0):
                caterror=dataset_dir+"test/cat/"+curfilename
                catorgfpath=datafile.img_list[n]
                            #print (caterror)
                shutil.copyfile(catorgfpath,caterror)
            else:
                caterror=dataset_dir+"test/dog/"+curfilename
                catorgfpath=datafile.img_list[n]
                            #print (caterror)
                shutil.copyfile(catorgfpath,caterror)
                     
    print(out.data)
    print("out1:"+str(out[0, 0])+" out2:"+str(out[0,1]))
    if out[0, 0] > out[0, 1]:
        print("the picture is a cat")
    else:
        print("the picture is a dog")
    img = Image.open(datafile.img_list[index])  # 打开对应索引的原始图像
    plt.figure('image')  # 创建一个新的图像窗口
    plt.imshow(img)  # 显示图片

    # 在图表上添加文本
    if out[0, 0] > out[0, 1]:
        plt.text(10, 10, 'this is a cat',color='red', size=12)
    else:
        plt.text(10, 10, 'this is a dog',color='red', size=12)
    # 添加标题
    plt.title("out1:"+"{:.4f}".format(out[0, 0])+" out2:"+"{:.4f}".format(out[0, 1]))
    
    plt.show()  # 显示图像窗口
if __name__ == "__main__":
   main()