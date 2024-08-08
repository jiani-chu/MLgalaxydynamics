#Jiani Chu 2022.10.17 update

import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import  torch.cuda as cuda
import datetime
import ResNet
import loaddata
import plot
import test

Name="name_of_target" ######

path='./'

Noutput=1
BATCHSIZE=40
nepoch = 200#100 # 200  # number of epochs
lr=0.0001

mufasa,mufasa_loader=loaddata.loaddata(Name,BATCHSIZE=BATCHSIZE)

#train/test dataset capture
train_size=16000
valid_size=4000
test_size=len(mufasa)-valid_size-train_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(mufasa, [train_size, valid_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=2)
#test
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)

# device = torch.device("cuda:0")
#
#neural network
net = ResNet.resnet18()# net = ResNet.resnet18()
net.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.01)#learning rate
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
steps = 200
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
# ##
print_num = 1000

train_loss, valid_loss = [],[]
tr_var_true,tr_var_predict=[],[]
vd_var_true,vd_var_predict=[],[]
tr_fid,tr_sid,vd_fid,vd_sid=[],[],[],[]
tr_proj,vd_proj=[],[]

for epoch in range(nepoch):  # loop over the dataset multiple times

    running_loss = 0.0
    # running_loss_1 = 0.0
    # running_loss_2 = 0.0
    loss_train=0.0
    # loss_train1=0.0
    # loss_train2=0.0

    net.train()
    predictions0, actuals0,galfid, galsid, galproj = list(), list(), list(), list(), list()
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        image1,image2,image3,galid, label,proj = data

        image1 = image1.cuda()
        image2 = image2.cuda()
        image3 = image3.cuda()
        # galid = galid.cuda()
        label = label.cuda()

        # proj = proj.cuda()


        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(image1,image2,image3)  # train with  images
        loss = criterion(outputs.float(),label.float())
        #         loss= loss.to(torch.float32)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        loss_train += loss.item()#loss in one batch

        if i % print_num == (print_num - 1):  # print every 10000 mini-batches
            print('[%d, %5d] loss: %.6f ' %
                  (epoch + 1, i + 1, running_loss / print_num))
            running_loss = 0.0

        outputs = outputs.cpu().detach().numpy()
        actual = label.cpu().numpy()
        galid = galid.numpy()
        proj = proj.numpy()


        predictions0.append(outputs)
        actuals0.append(actual)
        galfid.append(galid[:,0])
        galsid.append(galid[:,1])
        galproj.append(proj)


    train_loss.append(loss_train / len(train_loader))#loss in one epoch

    predictions0=np.array(predictions0)

    actuals0=np.array(actuals0)

    tr_var_true.append(actuals0)
    tr_var_predict.append(predictions0)

    galfid=np.array(galfid)
    galsid=np.array(galsid)
    galproj=np.array(galproj)

    tr_fid.append(galfid)
    tr_sid.append(galsid)
    tr_proj.append(galproj)


    ##save model
    dir = path+'models/' + Name + '_ResNet_epoch' + str(epoch) + '.pth'
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,'lr': scheduler.get_lr()}
    torch.save(state, dir)


    #eval model


    with torch.no_grad():
        net.eval()
        running_loss_validate = 0.0

        predictions0_vd, actuals0_vd, galfid_vd,galsid_vd,galproj_vd = list(), list(), list(), list(),list()
        for j, data2 in enumerate(valid_loader, 0):
            image_vd1,image_vd2,image_vd3, galid_vd, label_vd,proj_vd  = data2
            image_vd1 = image_vd1.cuda()#(10,1,48,48)#(batchsize,1,48,48)
            image_vd2 = image_vd2.cuda()
            image_vd3 = image_vd3.cuda()
            # galid_vd = galid_vd.cuda()
            # proj_vd = proj_vd.cuda()
            label_vd = label_vd.cuda()
            prediction_vd = net(image_vd1,image_vd2,image_vd3)
            loss_validate = criterion(prediction_vd.float(), label_vd.float())
            running_loss_validate += loss_validate.item()

            # to cpu data
            prediction_vd = prediction_vd.cpu().detach().numpy()#(10,1)#(batchsize,1)
            actual_vd = label_vd.cpu().numpy()
            galid_vd = galid_vd.numpy()
            proj_vd = proj_vd.numpy()

            predictions0_vd.append(prediction_vd)
            actuals0_vd.append(actual_vd)
            galfid_vd.append(galid_vd[:, 0])
            galsid_vd.append(galid_vd[:, 1])
            galproj_vd.append(proj_vd)

        actuals0_vd = np.array(actuals0_vd)#(400,10,1)#(batchsize,size/batchsize,1)
        predictions0_vd = np.array(predictions0_vd)
        vd_var_true.append(actuals0_vd)#all data at every epoch#(nepoch,batchsize,size/batchsize,1)
        vd_var_predict.append(predictions0_vd)#all data at every epoch
        galfid_vd = np.array(galfid_vd)
        galsid_vd = np.array(galsid_vd)
        galproj_vd = np.array(galproj_vd)
        vd_fid.append(galfid_vd)#all data at every epoch
        vd_sid.append(galsid_vd)#all data at every epoch
        vd_proj.append(galproj_vd)#all data at every epoch

    valid_loss.append(running_loss_validate / len(valid_loader))#validation loss in one epoch

    if epoch % 1 == 0:
        # plot.plot()
        plot.perform(actuals0,predictions0,actuals0_vd,predictions0_vd,Name,epoch)

    scheduler.step()
    print(scheduler.get_lr())
print('Finished Training')

#save
# dir='/work/militurbo/2022AUTUMNML/eval/'+Name+'_ResNet.pth'
# state = {'net':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
# torch.save(state, dir)

#loss curve
plot.losscurve(train_loss,valid_loss,Name,nepoch)

train_loss=np.array(train_loss)
valid_loss=np.array(valid_loss)
dict = {'TrainLoss':train_loss,'ValidLoss':valid_loss}
df = pd.DataFrame(dict)
df.to_csv(path+Name+'_loss.csv')#lossfile

#traing/validation set reslut at every epoch
tr_var_true=np.array(tr_var_true)
tr_var_predict=np.array(tr_var_predict)
vd_var_true=np.array(vd_var_true)
vd_var_predict=np.array(vd_var_predict)
data={'tr_var_true':tr_var_true,'tr_var_predict':tr_var_predict,'tr_fid':tr_fid,'tr_sid':tr_sid,'tr_proj':tr_proj}
with open(path+Name+'_label_tr.dat', 'wb') as f:
    pickle.dump(data, f)
data={'vd_var_true': vd_var_true, 'vd_var_predict': vd_var_predict,'vd_fid':vd_fid,'vd_sid':vd_sid,'vd_proj':vd_proj}
with open(path+Name+'_label_vd.dat', 'wb') as f:
    pickle.dump(data, f)

print('Done')

ind=int(np.array(np.where(valid_loss==np.min(valid_loss))))
test.testperform(test_loader,Name,ind,path,net,criterion,optimizer,scheduler)
