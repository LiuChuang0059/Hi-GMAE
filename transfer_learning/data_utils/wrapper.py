from torch.utils.data import Dataset



class Wrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.proj = []
        self.super_adj = []
        self.node_dict = []
        self.pe_list = []

    def put_item(self, item):
        self.proj = item[0]
        self.super_adj = item[1]
        self.node_dict = item[2]

    def put_pe(self, pe_list):
        self.pe_list = pe_list

    def __getitem__(self, index):
        data = self.dataset[index]
        data['proj'] = self.proj[index]
        data['super_adj'] = self.super_adj[index]
        data['node_dict'] = self.node_dict[index]
        data['pe'] = self.pe_list[index]
        return data

    def __len__(self):
        return len(self.dataset)


