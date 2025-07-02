import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from collections import defaultdict
import random
from torchvision.transforms import v2
import torch
transform=v2.Compose([
    v2.RandomAffine(degrees=40,translate=(10/229,0))
])
def file_path_generator(root_dir,subject_name,registration_form='df',image_type='dfT1'):
    image_type_name_dict={'dfT1':'wmi','dfGM':'mwp1','dfWM':'mwp2','ndfT1':'rmi','ndfGM':'rp1','ndfWM':'rp2'}
    data_folder_subject_name_dict={'IXI':r'C:\OpenData\IXI',
                                   'OAS1':r'C:\OpenData\OASIS',
                                   'ABIDE1':r'C:\OpenData\ABIDE 1',
                                   'ABIDE2':r'C:\OpenData\ABIDE 2',
                                   'FCON':r'C:\OpenData\FCP1000',
                                   'HCA':r'C:\OpenData\HCP-Aging',
                                   'SALD_Sub':r'C:\OpenData\SALD',
                                   'NKI-RS':r'C:\OpenData\NKI-RS',
                                   'CC':r'C:\OpenData\CC359',
                                   'sub_OAS3':r'C:\OpenData\OASIS3',
                                   }
    subject_foler_dict=['IXI','OAS1','ABIDE1','ABIDE2','FCON','HCA','SALD_Sub','NKI-RS','CC','sub_OAS3']
    
    for i in subject_foler_dict:
        if i in subject_name:
            root_dir=data_folder_subject_name_dict[i]
    if registration_form == 'df':
        lastwords='_raw_tbet.nii'
    elif registration_form == 'ndf':
        lastwords='_raw_tbet_rigid.nii' 

    return root_dir+'\\'+registration_form+'\\'+image_type_name_dict[image_type]+'\\'+image_type_name_dict[image_type]+subject_name+lastwords

class BrainMRIDataset(Dataset):
    def __init__(self, csv_file, root_dir, registration_form='df', image_type='dfT1', purpose='train',
                 train_size_per_epoch=210, val_size=45, test_size=45, transform=transform):

        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.registration_form = registration_form
        self.image_type = image_type
        self.purpose = purpose
        self.transform = transform

        self.train_size_per_epoch = train_size_per_epoch
        self.val_size = val_size
        self.test_size = test_size
    
        self.split_data()
        self.data_info = self.get_purpose_data()
        


        # 預先建立路徑
        self.data_info["image_path"] = self.data_info["Subject_ID"].apply(
            lambda x: file_path_generator(self.root_dir, x, self.registration_form, self.image_type)
        )
        
        


    def split_data(self):
        agelist = [(10 * p + 5, 10 * p + 15) for p in range(0, 9)]
        self.train_info, self.val_info, self.test_info = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for age_min, age_max in agelist:
            if age_min == 75:
                val_n, test_n = 25, 25
            elif age_min == 85:
                age_max, val_n, test_n = 105, 10, 10
            else:
                val_n, test_n = self.val_size, self.test_size

            subset = self.df.query(f"AGE>={age_min} & AGE<{age_max}")
            test_set = subset.sample(test_n, random_state=1)
            val_set = subset.drop(test_set.index).sample(val_n, random_state=1)
            train_set = subset.drop(test_set.index).drop(val_set.index)

            self.train_info = pd.concat([self.train_info, train_set])
            self.val_info = pd.concat([self.val_info, val_set])
            self.test_info = pd.concat([self.test_info, test_set])
    
    def train_data_sample(self):
        agelist = [(10 * p + 5, 10 * p + 15) for p in range(0, 9)]
        self.train_sampled=pd.DataFrame()

        for age_min, age_max in agelist:
            if age_min == 75:
                sample_n=120
            elif age_min == 85:
                age_max, sample_n = 105, 47
            else:
                sample_n=self.train_size_per_epoch

            subset = self.train_info.query(f"AGE>={age_min} & AGE<{age_max}")
            train_sample_set = subset.sample(sample_n)
            self.train_sampled = pd.concat([self.train_sampled, train_sample_set])
            
    def get_purpose_data(self):
        if self.purpose == 'train':
            self.train_data_sample()
            return self.train_sampled.reset_index(drop=True)
        elif self.purpose == 'val':
            return self.val_info.reset_index(drop=True)
        elif self.purpose == 'test':
            return self.test_info.reset_index(drop=True)
    
    
       
   


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        image_path = row["image_path"]
        
        try:
            image = loadMR(image_path)
            
            if self.purpose=='train':
                image = image.unsqueeze(0) if image.ndim == 3 else image
                image = self.transform(image)
            else :
                image = image.unsqueeze(0) if image.ndim == 3 else image
        except Exception as e:
            print(f"[Error] Failed loading idx {idx}: {e}")
            image = torch.zeros((1, 193, 229, 193), dtype=torch.float32)  # placeholder

        sample = {
            "image": image,
            "gender": torch.tensor(row["SEX(0=Male,1=Female)"], dtype=torch.float32),
            "age": torch.tensor(row["AGE"], dtype=torch.float32),
            "scanner": torch.tensor(row["Scanner"], dtype=torch.float32),
            "image_path":image_path
        }
        return sample


class BalancedAgeBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=4,dataset_type='train'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_type =dataset_type
        # 根據累計人數將年齡區間動態切成 batch_size 等分（依據年齡排序後的樣本）
        self.age_bins = self._compute_quantile_bins()
        self.age_to_indices = self._group_indices_by_age()
        self.batches=self._generate_balanced_batches()

    def _compute_quantile_bins(self):
        # 取得所有年齡值
        ages = self.dataset.data_info['AGE'].values
        # 計算分位數，將樣本依年齡分成 batch_size 等分
        quantiles = np.linspace(0, 1, self.batch_size + 1)
        bins = np.quantile(ages, quantiles).tolist()
        age_bins = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        # 為避免 bin 上下限重疊，微調最大邊界
        age_bins[-1] = (age_bins[-1][0], age_bins[-1][1] + 0.1)
        
        return age_bins

    def _group_indices_by_age(self):
        # 初始化每個年齡區間的 index 容器
        age_to_indices = {bin_range: [] for bin_range in self.age_bins}
        for idx in range(len(self.dataset)):
            age = float(self.dataset.data_info.loc[idx, 'AGE'])
            # 將每個 index 分配到對應的年齡區間中
            for bin_range in self.age_bins:
                if bin_range[0] <= age < bin_range[1]:
                    age_to_indices[bin_range].append(idx)
                    break
        return age_to_indices

    # def _generate_balanced_batches(self):
    #     batches = []
    #     # 隨機打亂每個年齡 bin 內的 index
    #     # bin_iters = {k: random.sample(v, len(v)) for k, v in self.age_to_indices.items() if len(v) > 0}
    #     bin_iters = {k: v for k, v in self.age_to_indices.items() if len(v) > 0}
        
    #     pointers = {k: 0 for k in bin_iters}
        
    #     # 動態產生 batch，每個 batch 包含不同年齡區間的樣本
    #     while True:
    #         batch = []
            
    #         for bin_range in self.age_bins:

    #             print(bin_range)
    #             if bin_range not in bin_iters or pointers[bin_range] >= len(bin_iters[bin_range]):
    #                 continue
    #             batch.append(bin_iters[bin_range][pointers[bin_range]])
    #             pointers[bin_range] += 1
                
    #             if len(batch) == self.batch_size:
    #                 break
    #         if len(batch) == self.batch_size:
    #             batches.append(batch)
    #         else:
    #             # 已無法構成完整 batch 時停止
               
    #             break
            
       
    #     return batches

    # def __iter__(self):
    #     for batch in self.batches:
    #         yield batch
    def _generate_balanced_batches(self):
        batches = []
        # 每個 bin 內資料先隨機 shuffle
        bin_iters = {k: v for k, v in self.age_to_indices.items() if len(v) > 0}
        pointers = {k: 0 for k in bin_iters}

        total_samples = sum(len(v) for v in bin_iters.values())
        samples_used = 0

        while samples_used < total_samples:
            batch = []
            # 每次從不同 bin 輪流取一筆
            for bin_range in self.age_bins:
                if bin_range not in bin_iters:
                    continue
                if pointers[bin_range] < len(bin_iters[bin_range]):
                    batch.append(bin_iters[bin_range][pointers[bin_range]])
                    pointers[bin_range] += 1
                if len(batch) == self.batch_size:
                    break

            if batch:  # 不管 batch 有幾筆，只要非空，就 append
                batches.append(batch)
                samples_used += len(batch)
            else:
                break  # 當所有 bin 都取光時停止

        return batches

    def __iter__(self):
        # 每個 epoch 重新 shuffle batch 順序
        if self.dataset_type=='train':
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
    def __len__(self):
        return len(self.batches)

def loadMR(path):
    try:
        img = nib.load(path).get_fdata().astype(np.float32).squeeze()
        return torch.tensor(img)
    except Exception as e:
        print(f"[loadMR] I/O Error at: {path}, Error: {e}")
        raise e


def swap_val_test_samples(dataset_val,dataset_test, val_idx, test_idx):
    # 取得兩邊資料
        val_row = dataset_val.data_info.iloc[val_idx].copy()
        test_row = dataset_test.data_info.iloc[test_idx].copy()

        # 交換
        dataset_val.data_info.iloc[val_idx] = test_row
        dataset_test.data_info.iloc[test_idx] = val_row

        dataset_val.val_info.reset_index(drop=True)
        dataset_test.val_info.reset_index(drop=True)
        
      
        print(f"Swapped val index {val_idx} with test index {test_idx}.")


if __name__ == '__main__':
    def swap_val_test_samples(dataset_val,dataset_test, val_idx, test_idx):
    # 取得兩邊資料
        val_row = dataset_val.data_info.iloc[val_idx].copy()
        test_row = dataset_test.data_info.iloc[test_idx].copy()

        # 交換
        dataset_val.data_info.iloc[val_idx] = test_row
        dataset_test.data_info.iloc[test_idx] = val_row

        dataset_val.val_info.reset_index(drop=True)
        dataset_test.val_info.reset_index(drop=True)
        
      
        print(f"Swapped val index {val_idx} with test index {test_idx}.")
    
    val_dataset = BrainMRIDataset(
        csv_file=r'C:\OpenData\DATA_LIST\data_list_for_train.csv',
        root_dir=r'C:\OpenData',
        registration_form='ndf',
        image_type='ndfT1',
        purpose='val',
        transform=transform
    )
    test_dataset = BrainMRIDataset(
        csv_file=r'C:\OpenData\DATA_LIST\data_list_for_train.csv',
        root_dir=r'C:\OpenData',
        registration_form='ndf',
        image_type='ndfT1',
        purpose='test',
        transform=transform
    )
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    val_sampler = BalancedAgeBatchSampler(test_dataset, batch_size=4,dataset_type='val')
    val_loader = DataLoader(test_dataset, batch_sampler=val_sampler,pin_memory=True)
    
    # print(val_dataset[349])
    # print(test_dataset[349])
   
    
    swap_val_test_samples(val_dataset,test_dataset, 349, 349)
    
    # print(val_dataset[349])   
    # print(test_dataset[349])
    
    # print(sum(dataset.data_info['AGE']>=100))
    # print((val_dataset.data_info['AGE']))
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     num_workers=0,  # 測試階段設為 0，避免多進程 I/O 卡住
    #     pin_memory=True,
    #     drop_last=False
    # )
    # age=[]

    for i, data in enumerate(val_loader):
        print(len(val_loader),i)
        inputs = data['image'].view(-1,1,193,229,193).to(device)
        print(torch.max(inputs),torch.min(inputs))
        labels = data['age'].unsqueeze(1).to(torch.float32)

        
        
    #     # print(f"Batch {i}: image shape {inputs.shape}, age: {labels.view(-1)}")
    # print(age)
