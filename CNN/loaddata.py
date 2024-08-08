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


Ncannot1=1e10# Vmean map dead pixel
Ncannot2=-1#V sigma map dead pixel
Ncannot3=1e10# r image dead pixel
Pmn1=-1100#min value of Vmean map pixel
Pmx1=1210#max value of Vmean map pixel
Pmn2=0#min value of Vsigma map pixel
Pmx2=1000#max value of Vsigma map pixel
Pmn3=-11#min value of r image  pixel
Pmx3=41#max value of r image pixel

#dataset
torch.manual_seed(1)# fixed split of traing& validation& test set

lumsun=10**(-0.4*4.65)#solar luminosity

#using Subfindid=FID+SID to match
file7 = 'MainSnapProps_099vol8.dat'
with open(file7, 'rb') as f:
    data = pickle.load(f, encoding='bytes')

fN = data['fN']
FID = data['FID']
SID = data['SID']
#<Reff
dm = data['DM_Re']#M_sun
tm = data['TM_Re']#M_sun
sm = data['SM_Re']#M_sun
sm_x = data['SM_Re_x']#M_sun
sm_y = data['SM_Re_y']#M_sun
sm_z = data['SM_Re_z']#M_sun
lum_x = data['Lum_Re_x']#10^{-0.4Mag}
lum_y = data['Lum_Re_y']#10^{-0.4Mag}
lum_z = data['Lum_Re_z']#10^{-0.4Mag}
#
mstar30 = data['Mstar_30kpc']#Msun
m200 = data['m200']#Msun
mstar_all = data['Mstar_all']#Msun

fN = np.array(fN)
FID = np.array(FID)
SID = np.array(SID)
Subfindid=FID+SID
projcting=np.ones_like(FID)# direction of projection 0,1,2
sm = np.array(sm)
dm = np.array(dm)
sm_x = np.array(sm_x)
sm_y = np.array(sm_y)
sm_z = np.array(sm_z)
lum_x =np.array(lum_x)
lum_y =np.array(lum_y)
lum_z =np.array(lum_z)

mstar30 = np.array(mstar30)
m200 = np.array(m200)
mstar_all = np.array(mstar_all)

# Re = data['HSMR']# kpc
# Re = np.array(Re)
# dm_p5re=data['DM_p5Re']#M_sun
# dm_2re=data['DM_2Re']#M_sun
# dm_p5re=np.array(dm_p5re)
# dm_2re=np.array(dm_2re)
# sm_p5re=data['SM_p5Re']#M_sun
# sm_p5re=np.array(sm_p5re)
# sm_2re=data['SM_2Re']#M_sun
# sm_2re=np.array(sm_2re)
# rhos=sm/Re**3
# rhot=tm/Re**3

m2l_x=sm_x/lum_x*lumsun
m2l_y=sm_y/lum_y*lumsun
m2l_z=sm_z/lum_z*lumsun

fdm=dm/tm

sm=np.log10(sm)
dm=np.log10(dm)
tm=np.log10(tm)


labels1={'fdm':fdm,'m2l':m2l_x,'mstar':sm,'mdm':dm,'mtot':tm}
labels2={'fdm':fdm,'m2l':m2l_y,'mstar':sm,'mdm':dm,'mtot':tm}
labels3={'fdm':fdm,'m2l':m2l_z,'mstar':sm,'mdm':dm,'mtot':tm}
LABELS1=pd.DataFrame(data=labels1)
LABELS2=pd.DataFrame(data=labels2)
LABELS3=pd.DataFrame(data=labels3)
# print(LABELS1['mstar'].shape)
# print(sm.shape)
def loaddata(labelkey,BATCHSIZE):
    # print(LABELS1[labelkey][0])
    LABEL1 = LABELS1[labelkey]
    LABEL2 = LABELS2[labelkey]
    LABEL3 = LABELS3[labelkey]
    LABEL1=np.array(LABEL1)
    LABEL2=np.array(LABEL2)
    LABEL3=np.array(LABEL3)
    mufasa = []
    num = 0
    for i in range(500):#10 for test
        file1 = '/work/militurbo/2021ML/DATASET' + '/Galaxy_f' + str(i) + '_iamges_NAT1.dat'#X projection
        file2 = '/work/militurbo/2021ML/DATASET' + '/Galaxy_f' + str(i) + '_iamges_NAT2.dat'#Y projection
        file3 = '/work/militurbo/2021ML/DATASET' + '/Galaxy_f' + str(i) + '_iamges_NAT3.dat'#Z projection
        file4 = '/work/militurbo/2021ML/DATASET' + '/Galaxy_f' + str(i) + '_labels.csv'#some labels: Fcold,Fhot...
        file6 = 'Galaxy_no_dark_ID.dat'#Subfinid id of galaxy which N_dm<1000, from Liang Yan

        try:

            f1 = open(file1, 'rb')
            data1 = pickle.load(f1)
            f2 = open(file2, 'rb')
            data2 = pickle.load(f2)
            f3 = open(file3, 'rb')
            data3 = pickle.load(f3)
            data4 = pd.read_csv(file4)

            data6 = np.loadtxt(file6)
            notid=data6### subfind id of galaxy which Ndm<1000

            fid = data4['FID']
            sid = data4['SID']
            fid = np.array(fid)
            sid = np.array(sid)
            GalID = np.zeros((len(fid), 2))# FID and SID of sample
            GalID[:, 0] = fid
            GalID[:, 1] = sid

            Fcold = data4['Fcold_Re']#cold orbit fraction
            Fcold = np.array(Fcold)


            Vmean1 = data1['MAPvmean']
            Vmean2 = data2['MAPvmean']
            Vmean3 = data3['MAPvmean']
            Vmean1 = np.reshape(Vmean1, (len(Fcold), 1, 48, 48))#(,,,)
            Vmean2 = np.reshape(Vmean2, (len(Fcold), 1, 48, 48))
            Vmean3 = np.reshape(Vmean3, (len(Fcold), 1, 48, 48))
            #deal with dead pixel
            ind1=np.where(Vmean1==Ncannot1)
            Vmean1[ind1]=0
            ind2 = np.where(Vmean2 == Ncannot1)
            Vmean2[ind2] = 0
            ind3 = np.where(Vmean3 == Ncannot1)
            Vmean3[ind3] = 0
            #normalization!!
            Vmean1 = (Vmean1 - Pmn1) / (Pmx1-Pmn1)
            Vmean2 = (Vmean2 - Pmn1) / (Pmx1-Pmn1)
            Vmean3 = (Vmean3 - Pmn1) / (Pmx1-Pmn1)

            Vsigma1 = data1['MAPvsigma']
            Vsigma2 = data2['MAPvsigma']
            Vsigma3 = data3['MAPvsigma']
            Vsigma1 = np.reshape(Vsigma1, (len(Fcold), 1, 48, 48))
            Vsigma2 = np.reshape(Vsigma2, (len(Fcold), 1, 48, 48))
            Vsigma3 = np.reshape(Vsigma3, (len(Fcold), 1, 48, 48))
            ind1 = np.where(Vsigma1 == Ncannot2)
            Vsigma1[ind1] = 0
            ind2 = np.where(Vsigma2 == Ncannot2)
            Vsigma2[ind2] = 0
            ind3 = np.where(Vsigma3 == Ncannot2)
            Vsigma3[ind3] = 0
            Vsigma1 = (Vsigma1 - Pmn2) / (Pmx2 - Pmn2)
            Vsigma2 = (Vsigma2 - Pmn2) / (Pmx2 - Pmn2)
            Vsigma3 = (Vsigma3 - Pmn2) / (Pmx2 - Pmn2)

            IMGr1 = data1['IMGr']
            IMGr2 = data2['IMGr']
            IMGr3 = data3['IMGr']
            IMGr1 = np.reshape(IMGr1, (len(Fcold), 1, 300, 300))
            IMGr2 = np.reshape(IMGr2, (len(Fcold), 1, 300, 300))
            IMGr3 = np.reshape(IMGr3, (len(Fcold), 1, 300, 300))
            IMGr1 = (IMGr1 - Pmn3) / (Pmx3 - Pmn3)
            IMGr2 = (IMGr2 - Pmn3) / (Pmx3 - Pmn3)
            IMGr3 = (IMGr3 - Pmn3) / (Pmx3 - Pmn3)


            images11,images12,images13,images21,images22,images23,images31,images32,images33,galids,label1,label2,label3,proj1,proj2,proj3=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
            for j in range(len(Fcold)):
                if dm[np.where(Subfindid==GalID[j, 0]+GalID[j, 1])]!=0 and mstar30[np.where(Subfindid==GalID[j, 0]+GalID[j, 1])]>5e9 \
                        and m200[np.where(Subfindid==GalID[j, 0]+GalID[j, 1])]<1e14 and not(np.isin(GalID[j, 0]+GalID[j, 1],notid)):

                    #N_dm>1000 and M*>5e9 Mtot<1e14
                    num=num+1

                    images11.append(Vmean1[j, :])# ij, j==direction, i==maps
                    images12.append(Vmean2[j, :])
                    images13.append(Vmean3[j, :])
                    images21.append(Vsigma1[j, :])
                    images22.append(Vsigma2[j, :])
                    images23.append(Vsigma3[j, :])
                    images31.append(IMGr1[j, :])
                    images32.append(IMGr2[j, :])
                    images33.append(IMGr3[j, :])
                    galids.append(GalID[j,:])

                    label1.append(LABEL1[np.where(Subfindid == fid[j] + sid[j])])#M*/L(2D) depends on direction
                    label2.append(LABEL2[np.where(Subfindid == fid[j] + sid[j])])
                    label3.append(LABEL3[np.where(Subfindid == fid[j] + sid[j])])
                    proj1.append(projcting[np.where(Subfindid == fid[j] + sid[j])])#projection direction
                    proj2.append(2*projcting[np.where(Subfindid == fid[j] + sid[j])])
                    proj3.append(3*projcting[np.where(Subfindid == fid[j] + sid[j])])

                    # m2l1 = sm_p5re[np.where(Subfindid == fid[j] + sid[j])] / lum_p5re[np.where(Subfindid == fid[j] + sid[j])] * lumsun
                    # m2l2 = sm_2re[np.where(Subfindid == fid[j] + sid[j])] / lum_2re[np.where(Subfindid == fid[j] + sid[j])] * lumsun
                    # r1 = 0.5 * Re[np.where(Subfindid == fid[j] + sid[j])]
                    # r2 = 2 * Re[np.where(Subfindid == fid[j] + sid[j])]
                    # label.append((np.log10(m2l1) - np.log10(m2l2)) / (np.log10(r1) - np.log10(r2)))

            images11 = np.array(images11)
            images12 = np.array(images12)
            images13 = np.array(images13)
            images21 = np.array(images21)
            images22 = np.array(images22)
            images23 = np.array(images23)
            images31 = np.array(images31)
            images32 = np.array(images32)
            images33 = np.array(images33)
            label1=np.array(label1)
            label2 = np.array(label2)
            label3 = np.array(label3)
            proj1 = np.array(proj1)
            proj2 = np.array(proj2)
            proj3 = np.array(proj3)
            galids=np.array(galids)


            images11 = torch.from_numpy(images11)
            images12 = torch.from_numpy(images12)
            images13 = torch.from_numpy(images13)
            images21 = torch.from_numpy(images21)
            images22 = torch.from_numpy(images22)
            images23 = torch.from_numpy(images23)
            images31 = torch.from_numpy(images31)
            images32 = torch.from_numpy(images32)
            images33 = torch.from_numpy(images33)

            labels1 = torch.from_numpy(label1)
            labels2 = torch.from_numpy(label2)
            labels3 = torch.from_numpy(label3)
            Proj1 = torch.from_numpy(proj1)
            Proj2 = torch.from_numpy(proj2)
            Proj3 = torch.from_numpy(proj3)
            GalIDs = torch.from_numpy(galids)

            mufasa.append(TensorDataset(images11, images21, images31, GalIDs, labels1,Proj1))
            mufasa.append(TensorDataset(images12, images22, images32, GalIDs, labels2,Proj2))
            mufasa.append(TensorDataset(images13, images23, images33, GalIDs, labels3,Proj3))

        except:
            continue

    mufasa=ConcatDataset(mufasa)
    mufasa_loader = DataLoader(mufasa, batch_size=BATCHSIZE, shuffle=True)
    # print(len(mufasa))
    # print(len(mufasa_loader))
    # print(num)

    return mufasa,mufasa_loader