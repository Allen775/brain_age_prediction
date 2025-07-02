import torch 
import numpy
from torchsummary import summary
from new_dataloader import BrainMRIDataset
from new_dataloader import BalancedAgeBatchSampler
from new_dataloader import swap_val_test_samples
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime  
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import gc
class ResBlock(torch.nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv1=torch.nn.Conv3d(in_channels,out_channels,3,1,1)
        self.batchnorm=torch.nn.BatchNorm3d(out_channels,affine=True,track_running_stats=False)
        self.ELU=torch.nn.ELU()
        self.conv2=torch.nn.Conv3d(out_channels,out_channels,3,1,1)
        self.conv_shortcut=torch.nn.Conv3d(in_channels,out_channels,1,1,0)
        self.maxpooling=torch.nn.MaxPool3d(2,2,ceil_mode=False)
    def forward(self, x):
        
        x_=self.conv_shortcut(x)
        x=self.conv1(x)
        x=self.batchnorm(x)
        x=self.ELU(x)
        x=self.conv2(x)
        x=self.batchnorm(x)
        x=self.ELU(x+x_)
        return x
    
class NetResBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resblock1=ResBlock(1,8)
        self.resblock2=ResBlock(8,16)
        self.resblock3=ResBlock(16,32)
        self.resblock4=ResBlock(32,64)
        self.resblock5=ResBlock(64,128)
        self.resblock6=ResBlock(128,256)
        self.maxpooling1=torch.nn.MaxPool3d(2,2,1,ceil_mode=False)
        self.maxpooling2=torch.nn.MaxPool3d(2,2,(1,0,1),ceil_mode=False)
        
    def forward(self, x):
        x=self.resblock1(x)
        x=self.maxpooling1(x)
        x=self.resblock2(x)
        x=self.maxpooling1(x)
        x=self.resblock3(x)
        x=self.maxpooling2(x)
        x=self.resblock4(x)
        x=self.maxpooling1(x)
        x=self.resblock5(x)
        x=self.maxpooling1(x)
        x=self.resblock6(x)
        x=self.maxpooling2(x)
        return x 
    
class RESNET_MODEL(torch.nn.Module):
    def __init__(self,model_type):
        super().__init__()
        self.RES=NetResBlock()
        self.flatten=torch.nn.Flatten()
        self.fc1=torch.nn.Linear(16384,256)
        self.ELU=torch.nn.ELU()
        self.dropout=torch.nn.Dropout(p=0.2)  
        self.fc2_for_no_cat=torch.nn.Linear(256,1)
        # self.fc2_for_cat=torch.nn.Linear(257,1)
        self.fc2_for_cat=torch.nn.Linear(258,1)
        self.model_type=model_type
        
    def forward(self,x,gender,scanner):
        
        x=self.RES(x)
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.ELU(x)
        x=self.dropout(x)
        if self.model_type == 'cat':
            x=torch.cat((x,gender,scanner),dim=1)
            # x=torch.cat((x,scanner),dim=1)
            x=self.fc2_for_cat(x)
        else:
            x=self.fc2_for_no_cat(x)
        return x



        
if __name__== '__main__':
    #parameters initialization 
    batch_size=4
    csv_file_path=r'C:\OpenData\DATA_LIST\data_list_for_train.csv'
    root_dir_path=r'C:\OpenData'
    registration='ndf'
    image_type='ndfT1'
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model_type='cat'

    saved_model_path=r"E:\pytorch_brainage\trained_model\model_20250628_092154_757.pt"
    
    #model initialization 
    pretrained_model = torch.load(saved_model_path,weights_only=False)
    model = RESNET_MODEL(model_type).to(device)
    model.load_state_dict(pretrained_model.state_dict(),strict=False)
    
    
    # model=RESNET_MODEL(model_type)
    # model=(torch.load(saved_model_path,weights_only=False))
    
    model.eval()
    
    
    #LOSS L1loss == MSE
    loss_fn=torch.nn.L1Loss()
    
    #dataset and data laoder 
    # train_dataset=BrainMRIDataset(csv_file=csv_file_path,root_dir=root_dir_path,registration_form=registration,image_type=image_type,purpose='train')
    # training_loader=DataLoader(train_dataset,batch_size,shuffle=True,num_workers=1,pin_memory=True,drop_last=True)

    val_dataset = BrainMRIDataset(
        csv_file=r'C:\OpenData\DATA_LIST\data_list_for_train.csv',
        root_dir=r'C:\OpenData',
        registration_form='ndf',
        image_type='ndfT1',
        purpose='val',
    )
    val_sampler = BalancedAgeBatchSampler(val_dataset, batch_size=4,dataset_type='val')
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,pin_memory=True)

    test_dataset = BrainMRIDataset(
        csv_file=r'C:\OpenData\DATA_LIST\data_list_for_train.csv',
        root_dir=r'C:\OpenData',
        registration_form='ndf',
        image_type='ndfT1',
        purpose='test',
    )
    test_sampler = BalancedAgeBatchSampler(test_dataset, batch_size=4,dataset_type='test')
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler,pin_memory=True)

    swap_val_test_samples(val_dataset,test_dataset,349,349)
    
    # testing_dataset=BrainMRIDataset(csv_file=csv_file_path,root_dir=root_dir_path,registration_form=registration,image_type=image_type,purpose='test')
    # testing_loader=DataLoader(testing_dataset,batch_size,shuffle=False,num_workers=1,pin_memory=True,drop_last=False)
    
    # print('testing data len: ',len(testing_loader))
    running_vloss=0.
    running_tloss=0.
    model.eval()
    with torch.no_grad():
            for i,vdata in enumerate(test_loader):
                inputs= vdata['image'].view(-1,1,193,229,193).to(device)
                labels=vdata['age'].view(-1,1).to(device)
                gender=((torch.t(vdata['gender'].unsqueeze(0))).int()).to(device)
                scanner=((torch.t(vdata['scanner'].unsqueeze(0))).int()).to(device)
                outputs = model(inputs,gender,scanner)
                tloss = loss_fn(outputs, labels)
                running_tloss+=tloss.item()  
            avg_tloss=running_tloss/(i+1)

            for i,vdata in enumerate(val_loader):
                inputs= vdata['image'].view(-1,1,193,229,193).to(device)
                labels=vdata['age'].view(-1,1).to(device)
                gender=((torch.t(vdata['gender'].unsqueeze(0))).int()).to(device)
                scanner=((torch.t(vdata['scanner'].unsqueeze(0))).int()).to(device)
                outputs = model(inputs,gender,scanner)
                vloss = loss_fn(outputs, labels)
                running_vloss+=vloss.item()
            avg_vloss=running_vloss/(i+1)
    
    print('validation loss {: .3f}'.format(avg_vloss))
    print('testing    loss {: .3f}'.format(avg_tloss))
      
      

    

    
    