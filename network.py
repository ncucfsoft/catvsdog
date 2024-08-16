import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self=self.to(device)
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.fc1 = torch.nn.Linear(50*50*16, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 2)
       
        #print(next(self.parameters()).is_cuda)
    def forward(self, x):
        # 定义前向传播过程        def forward(self, x):
        x = self.conv1(x)   # 第一层卷积
        x = F.relu(x)       # 激活函数 ReLU
        x = F.max_pool2d(x, 2)  # 池化层，大小为2x2的最大池化
        
        x = self.conv2(x)   # 第二层卷积
        x = F.relu(x)       # 激活函数 ReLU
        x = F.max_pool2d(x, 2)  # 池化层，大小为2x2的最大池化
        
        x = x.view(x.size(0), -1)  # 将特征图展平成一维向量
        
        x = F.relu(self.fc1(x))   # 第一个全连接层，并使用ReLU激活函数
        x = F.relu(self.fc2(x))   # 第二个全连接层，并使用ReLU激活函数
        x = self.fc3(x)           # 第三个全连接层，输出未经过激活函数
        #print(next(self.parameters()).is_cuda)
        return F.softmax(x, dim=1)  # 使用softmax函数进行分类，dim=1表示按行计算softmax
