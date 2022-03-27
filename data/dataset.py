import torch
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode):
        self.transform = transform
        self.mode = mode
    
        self.dlist = []


    def __len__(self):
        return len(self.dlist)
    
    
    def __getitem__(self, idx):    
        inputs = {
                
                }
        targets = {
                
                }
        meta = {
                
                }

        return inputs, targets, meta


