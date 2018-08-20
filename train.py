import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pickle
import numpy as np
from data import dataset
from models import model
from torch import optim
from config import DefaultConfig
import torch.nn.functional as F
import copy
# from tqdm import tqdm

config = DefaultConfig()

if config.use_hyperboard:
    from hyperboard import Agent
    agent = Agent(username='jlb', password='123',port=5005)
    parameter = config.todict()
    validate_loss_record = agent.register(parameter,'loss',overwrite=True)

train_dataset = dataset.MyDataset()
validate_dataset = dataset.MyDataset()

criticer = torch.nn.MSELoss()
model = model.Model()
optimizer = optim.Adam(model.parameters(),lr=config.lr)
if config.gpu>=0:
    model.cuda(config.gpu)

max_loss = 0
no_gain = 0
global_step = 0
train_num = len(train_dataset)

model.train()
for epoch in range(config.epoch_num):
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    for step,(traindata,trainlabel) in enumerate(train_loader):
        traindata = Variable(traindata).float()
        trainlabel = Variable(trainlabel).float()
        if config.gpu>=0:
            traindata = traindata.cuda(config.gpu)
            trainlabel = trainlabel.cuda(config.gpu)
        pred = model(traindata)
        loss = criticer(pred,trainlabel)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # validate
        if global_step % config.validate_step == 0:
            model.eval()
            validate_loader = DataLoader(validate_dataset, batch_size=config.validate_batch_size, shuffle=True)
            vloss = 0
            for vs,(vd,vl) in enumerate(validate_loader):
                vd = Variable(vd,volatile=True).float()
                vl = Variable(vl,volatile=True).float()
                if config.gpu>=0:
                    vd = vd.cuda(config.gpu)
                    vl = vl.cuda(config.gpu)
                vp = model(vd)
                vloss += criticer(vp,vl).data[0]
                if vs == config.validate_batch_num-1:
                    break
            vloss = vloss/config.validate_batch_num
            model.train()
            print('Epoch{} Step{}: [{}/{} ({:.0f}%)]\tValidate Loss: {:.6f}'.format(epoch, step, step*config.batch_size, train_num, 100.*step*config.batch_size/train_num,vloss))
            if config.use_hyperboard:
                agent.append(validate_loss_record,global_step,vloss)

            # early stop
            if config.early_stop_num >= 0:
                if vloss > max_loss:
                    max_loss = vloss
                    no_gain = 0
                else:
                    no_gain += 1
                if no_gain >= config.early_stop_num:
                    print('Early Stop!')
                    # save model
                    model_save = copy.deepcopy(model)
                    torch.save(model_save.cpu(), './checkpoint/early_model_{}epoch_{}step.mod'.format(epoch,step))
                    break

        # save model
        if step % config.save_step == 0:
            model_save = copy.deepcopy(model)
            torch.save(model_save.cpu(), 'model_{}epoch_{}step.mod'.format(epoch,step))

    # if config.early_stop_num >= 0:
    #     if no_gain >= config.early_stop_num:
    #         break









