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

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

img_size = 200 #设置图片尺寸

dataset_dir = 'd:/python/validation/'  # 数据集路径
model_dir = 'd:/python/validation/model/'     # 网络参数保存位置
workers = 10                        # 线程数量
batch_size = 50                     # 一次训练所选取的样本数
lr = 0.001                         # 学习率
nepoch = 1                         # 训练的次数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x),data_transforms[x]) for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=4) for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes
#tran = Trans.Compose([Trans.Resize(img_size), Trans.CenterCrop([img_size, img_size]), Trans.ToTensor()]) #封装， 对后面读取的图片的格式转换
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        # Create a temporary directory to save training checkpoints
        #with TemporaryDirectory() as tempdir:
        Bdatadir=True

    
        if(Bdatadir):
    
        
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

        return model


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

                        # zero the parameter gradients
                        optimizer_conv.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model_conv(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                            #print ('labels:'+str(labels)+'preds:'+str(preds))
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

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        #except:
          # print('fail img inload-----')   
           
    torch.save(model_conv.state_dict(), '{0}/model.pth'.format(model_dir))  # 保存训练好的模型参数

if __name__ == "__main__":
    train()