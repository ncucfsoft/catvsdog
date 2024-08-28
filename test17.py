import torch.utils.data
import os   #文件操作模块
import torch
import torch.utils.data as data   #用于继承一个父类（data.Dataset）里的函数
from PIL import Image
import torchvision.transforms as Trans #在定义图片转换的格式时，会用到相关的函数
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader as DataLoader
import time
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import tkinter as tk
from tkinter import messagebox
import shutil
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

img_size = 200 #设置图片尺寸

dataset_dir = 'd:/python/validation/'  # 数据集路径
model_file = 'd:/python/validation/model/model.pth'     # 网络参数保存位置
workers = 10                        # 线程数量
batch_size = 25                     # 一次训练所选取的样本数
lr = 0.001                         # 学习率
nepoch = 30                         # 训练的次数
image_datasets={}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_datasets['train']=datasets.ImageFolder(root=os.path.join(dataset_dir, 'train'),transform=data_transforms['train'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=4) for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes
#tran = Trans.Compose([Trans.Resize(img_size), Trans.CenterCrop([img_size, img_size]), Trans.ToTensor()]) #封装， 对后面读取的图片的格式转换


def train():
   

   
  
    print('Dataset loaded! length of train set is {0}'.format(len(dataloaders['train'])))
   
    model_conv=torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    #model = nn.DataParallel(model)  # 多GPU并行化
    #modle=model.to('cuda')
    model_conv.train()  # 将模型设置为训练模式
    print(torch.cuda.current_device())
    print(next(model_conv.parameters()).is_cuda)

   
    best_acc = 0.0

    for epoch in range(nepoch):
                print(f'Epoch {epoch}/{nepoch - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train']:
                    if phase == 'train':
                        model_conv.train()  # Set model to training mode
                    else:
                        model_conv.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        #print ('zero前:'+str(inputs)+'labels:'+str(labels))
                        # zero the parameter gradients
                        optimizer_conv.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model_conv(inputs)
                            _, preds = torch.max(outputs, 1)
                            #print ('criter前:'+str(labels)+'out:'+str(outputs))
                            loss = criterion(outputs, labels)
                           
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer_conv.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == 'train':
                        exp_lr_scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]
                    resultct=f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'
                    print(resultct)

        #except:
          # print('fail img inload-----')   
           
    torch.save(model_conv.state_dict(), '{0}'.format(model_file))  # 保存训练好的模型参数
    messagebox.showinfo('训练完毕!',resultct )
def regandcopy():
        
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    workers = 10                        # 线程数量
                        # 一次训练所选取的样本数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    bShowacc=False
    bShowSingleImg=False
    #image_datasets={}

    class_names=('cat','dog')

    model= models.resnet18(weights='IMAGENET1K_V1')
        #print(str(model))
    for param in model.parameters():
        param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

        #model = nn.DataParallel(model)  # 多GPU并行化
    model = model.to(device)
    model.load_state_dict(torch.load(model_file,weights_only=True))  # 加载模型参数
    model.eval()  # 将模型设置为评估模式
        #dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False)  # 创建数据加载器
    criterion = nn.CrossEntropyLoss()
        
        #index = np.random.randint(0, datafile.data_size, 1)[0]  # 随机选择一个索引
        #img = datafile.__getitem__(index).to(device)  # 获取对应索引的图片及其标签
    
    model.eval()

    

    CopycatdogFile(model,criterion)
    messagebox.showinfo('识别和分类图片完毕!', '识别结束，请查看图片自动归档到不同文件夹下的准确率！')
        
     
def enumerate_files(directory):
 
# 列出目录下的所有文件和文件夹
    entries = os.listdir(directory)
    #print (str(entries))
    # 过滤出文件
    files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]
    
    return files
def CopycatdogFile(model,criterion):
        
        dataset_test='D:/python/validation/test/'
       
        for file_path in enumerate_files(dataset_test):
            curfilepath=os.path.join(dataset_test, file_path)
            #print(curfilepath)
            img = Image.open(curfilepath).convert('RGB')
            img = data_transforms['test'](img)
            img = img.unsqueeze(0)
            img = img.to(device)

            with torch.no_grad():
                outputs = model(img)
                _, preds = torch.max(outputs, 1)

            curclassname=class_names[preds[0]]    
            #print(curclassname)
            #print("preds:"+str(preds))
            
            #print("the picture is a "+curclassname)
            curfilename=curfilepath.split(sep='/')[-1]

            caterror=dataset_test+"/"+curclassname+"/"+curfilename
            catorgfpath=curfilepath
              
         
                            #print (caterror)
            shutil.copyfile(catorgfpath,caterror)
           

if __name__ == "__main__":
    root = tk.Tk()
    tk.Button(root, text="开始训练", command=train).grid(row=2, column=1, sticky="w", padx=10, pady=5)
    tk.Button(root, text="开始自动识别和归类到文件夹", command=regandcopy).grid(row=2, column=2, sticky="e", padx=10, pady=5)
    width=500
    height=200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
    root.title("选择开始训练或者开始识别并归类图片")

    root.resizable(False,False)
    root.mainloop()