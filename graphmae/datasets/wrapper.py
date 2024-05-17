from torch.utils.data import Dataset


class Wrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.proj = []
        self.super_adj = []
        self.super_feature = []
        self.node_dict = []
        self.pe = []

    def put_item(self, item):
        self.proj = item[0]
        self.super_adj = item[1]
        self.super_feature = item[2]
        self.node_dict = item[3]
        self.pe = item[4]

    def __getitem__(self, index):
        data = self.dataset[index]
        data['proj'] = self.proj[index]
        data['super_adj'] = self.super_adj[index]
        data['super_feature'] = self.super_feature[index]
        data['node_dict'] = self.node_dict[index]
        data['pe'] = self.pe[index]
        return data

    def __len__(self):
        return len(self.dataset)
