
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
import cv2
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader as DataLoader
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

Bcopyrecognize=False#批量识别猫和狗图片，并把它们自动复制到cat和dog目录下。方便肉眼看有没有识别错的
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止打印图片出错
dataset_dir = 'd:/python/validation/'  # 测试集路径
model_file = 'd:/python/validation/model/model.pth'  # 模型文件路径
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
workers = 10                        # 线程数量
batch_size = 50                     # 一次训练所选取的样本数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
bShowacc=False
bShowSingleImg=False
#image_datasets={}

class_names=('cat','dog')
if(bShowacc or bShowSingleImg):
        image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x),data_transforms[x]) for x in ['test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=4) for x in ['test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
        class_names = image_datasets['test'].classes

       
     
def ShowtestAcc(model,criterion):
                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders['test']:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                    

                        # forward
                        # track history if only in train
                      
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                           

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                   

                    epoch_loss = running_loss / dataset_sizes['test']
                    epoch_acc = running_corrects.double() / dataset_sizes['test']

                    print(f'{'test'} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

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
            print(curfilepath)
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
           

def ShowSingleImg(model):
    img_path=''
    catpathxml= 'd:/python/validation/model/data/haarcascade_frontalcatface.xml'

    cat_cascade = cv2.CascadeClassifier(catpathxml)
    #print(str(cat_cascade))
    dog_cascade = cv2.CascadeClassifier('d:/python/validation/model/data/haarcascade_frontalface_alt2.xml')

 
    index = np.random.randint(0, len(image_datasets['test'].imgs), 1)[0]  # 随机选择一个索引
    #print (str(len(image_datasets['test'].imgs)))
    #print(str(image_datasets['test'].imgs))
    img_path=image_datasets['test'].imgs.__getitem__(index)[0]
    print (img_path)
    #img_path='D:/python/validation/test/2.jpg'
    #img = datafile.__getitem__(index).to(device)  # 获取对应索引的图片及其标签
 
    img = Image.open(img_path).convert('RGB')
    img = data_transforms['test'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        out, preds = torch.max(outputs, 1)

        
    #print(outputs.data)
    print("preds:"+str(preds))
    
    print("the picture is a "+class_names[preds[0]])
    
    img = Image.open(img_path)  # 打开对应索引的原始图像
    #img=img.resize((244,244))
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    fig, ax = plt.subplots(1)
    #fig=plt.figure('image')  # 创建一个新的图像窗口
    ax.imshow(img)
    if class_names[preds[0]]=='cat':
        cats = cat_cascade.detectMultiScale(img_cv2, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))#
        #print('cats:'+str(cats))
        for (x, y, w, h) in cats:
            #print('find xscale yscale:'+ax.()+ax.get_xticks())
            #plt.text(x, y, 'find cat face',color='red', size=12)
            #trans = ax.transData.transform((x + w, y + h))
            rect = plt.Rectangle(xy=(x, y), width=w, height=h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            #ax.set_axis_off()


            #plt.Rectangle( [x, y], width=x + w,height= y + h,color='red')
            #cv2.rectangle(img_cv2, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色矩形框
    else:
        dogs = dog_cascade.detectMultiScale(img_cv2, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))#
        #print('dogs:'+str(dogs))
        for (x, y, w, h) in dogs:
            rect = plt.Rectangle(xy=(x, y), width=w, height=h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
           # print('find dogs face:'+str(x))
           # plt.text(x, y, 'find dog face',color='red', size=12)
           # plt.Rectangle( [x, y],  width=x + w,height= y + h, color='red',)
            #cv2.rectangle(img_cv2, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色矩形框
    #img_cv2.imshow("",img)  # 显示图片
    text='this is a '+class_names[preds[0]]
    #cv2.putText(img_cv2,text,(10,20),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255, 0),1)
   
    # 展示图片并保存
    #cv2.imshow("Output", img_cv2)
    #cv2.waitKey(0)
    #img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # 在图表上添加文本
    
    plt.text(10, 10, 'this is a '+class_names[preds[0]],color='red', size=12)
    # 添加标题
    plt.title("out1:"+"{:.4f}".format(outputs[0, 0])+" out2:"+"{:.4f}".format(outputs[0, 1]))
    
    plt.show()  # 显示图像窗口
    
     
def main():
    #N = 10  # 随机选择测试图片的数量
    #print('image_datasets loaded! length of train set is {0}'.format(len(image_datasets['test'])))

    #print('dataloaders loaded! length of train set is {0}'.format(len(dataloaders['test'])))

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

   

   
    if(bShowacc):
     ShowtestAcc(model,criterion)

    bCopyfiletocatdog=True
    if(bCopyfiletocatdog):
      CopycatdogFile(model,criterion)

   
    if(bShowSingleImg):
        ShowSingleImg(model)

if __name__ == "__main__":
   main()