import torch 
from torchsummary import summary
from new_dataloader import BrainMRIDataset
from new_dataloader import BalancedAgeBatchSampler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime  
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



class ResBlock(torch.nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv1=torch.nn.Conv3d(in_channels,out_channels,3,1,'same')
        self.batchnorm=torch.nn.BatchNorm3d(out_channels,affine=True,track_running_stats=False)
        # self.batchnorm=torch.nn.InstanceNorm3d(out_channels,affine=True)
        self.ELU=torch.nn.ELU()
        self.conv2=torch.nn.Conv3d(out_channels,out_channels,3,1,'same')
        self.conv_shortcut=torch.nn.Conv3d(in_channels,out_channels,1,1,0)
        
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
            x=self.fc2_for_cat(x)
        else:
            x=self.fc2_for_no_cat(x)
        return x


        
        
        
if __name__== '__main__':
    #parameters initialization 
    timestamp= datetime.now().strftime('%Y%m%d_%H%M%S')
    EPOCHS=1000
    batch_size=4
    epoch_number=0
    csv_file_path=r'C:\OpenData\DATA_LIST\data_list_for_train.csv'
    root_dir_path=r'C:\OpenData'
    registration='ndf'
    image_type='ndfT1'
    best_vloss=1000000
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model_type='cat'

   

    #model initialization 
    model=RESNET_MODEL(model_type).to(device)

    #opimizer initialization
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-5)

    #LOSS L1loss == MAE
    loss_fn=torch.nn.L1Loss()
    
     
   

   

    val_dataset = BrainMRIDataset(
        csv_file=r'C:\OpenData\DATA_LIST\data_list_for_train.csv',
        root_dir=r'C:\OpenData',
        registration_form='ndf',
        image_type='ndfT1',
        purpose='val',
    )
    val_sampler = BalancedAgeBatchSampler(val_dataset, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,pin_memory=True)
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True
    # )

   
    #tensorboard writer 
    tb_writer=SummaryWriter('runs/brainage_trainer_{}'.format(timestamp))
    
    #training/val loops 
    for _ in range(EPOCHS):
        print('EPOCH{}'.format(epoch_number+1))
        model.train()
        running_loss = 0.
        last_loss = 0.
        
        train_dataset = BrainMRIDataset(
        csv_file=r'C:\OpenData\DATA_LIST\data_list_for_train.csv',
        root_dir=r'C:\OpenData',
        registration_form='ndf',
        image_type='ndfT1',
        purpose='train',
    )
        sampler = BalancedAgeBatchSampler(train_dataset, batch_size=4)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler,pin_memory=True)
    

        for i, data in enumerate(train_loader):
            
            # Every data instance is an input + label pair
            inputs = data['image'].view(-1,1,193,229,193).to(device)
            labels = data['age'].view(-1,1).to(device)
            gender = data['gender'].view(-1,1).to(torch.float32).to(device)
            scanner = data['scanner'].view(-1,1).to(torch.float32).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs,gender,scanner)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
                   
        last_loss = running_loss / (i+1) 

        
        running_vloss=0.
        avg_vloss=0.
        model.eval()
        with torch.no_grad():
            for i,vdata in enumerate(val_loader):
                inputs = vdata['image'].view(-1,1,193,229,193).to(device)
                labels = vdata['age'].view(-1,1).to(device)
                gender = vdata['gender'].view(-1,1).to(torch.float32).to(device)
                scanner = vdata['scanner'].view(-1,1).to(torch.float32).to(device)

                outputs = model(inputs,gender,scanner)
                vloss = loss_fn(outputs, labels)
                running_vloss+=vloss.item()
                      
        avg_vloss=running_vloss/(i+1)
        
        print('Training loss {: .3f}  validation loss {: .3f}'.format(last_loss, avg_vloss))
               
        tb_writer.add_scalars('Training vs. Validation Loss',
                             { 'Training' : last_loss, 'Validation' : avg_vloss },
                                epoch_number + 1)
        tb_writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'E:/pytorch_brainage/trained_model/model_{}_{}.pt'.format(timestamp, epoch_number+1)
            print("epoch {} model saved!".format(epoch_number+1))
            torch.save(model, model_path)
        elif (epoch_number+1)%100==0:
            model_path = 'E:/pytorch_brainage/trained_model/model_{}_{}.pt'.format(timestamp, epoch_number+1)
            print("epoch {} model saved!".format(epoch_number+1))
            torch.save(model, model_path)
        epoch_number += 1

    

    
    