from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, STL10,\
                                 VisionDataset, VOCSegmentation

import random
import numpy as np
import torch
from math import ceil
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from utils.augmentation import get_transform, MnistNorm
# Dataset 상속
class CustomDataset(Dataset): 
  def __init__(self,x_data,y_data):
    self.x_data = x_data
    self.y_data = y_data

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = self.x_data[idx]
    y = self.y_data[idx]
    return x, y


def _get_split_indices_cls(trainset, p, seed):
    train_targets = [components[1] for components in trainset]
    splitter = StratifiedShuffleSplit(1, test_size=p, random_state=seed) # straified only once.
    indices = range(len(trainset))
    return next(splitter.split(indices, train_targets)) # return train test index 


def _get_split_indices_rgs(trainset, p, seed):
    length = len(trainset)
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    sep = int(length * p)
    return indices[sep:], indices[:sep]


def _get_kfolded_indices_cls(valid_indices, trainset, num_k, seed):
    train_targets = [label for img, label in trainset]
    valid_targets = [train_targets[i] for i in valid_indices]
    splitter = StratifiedKFold(num_k, True, seed)
    mask_iter = splitter._iter_test_masks(valid_indices, valid_targets)
    kfolded_indices = [np.array(valid_indices[np.nonzero(m)]) for m in mask_iter]
    base_len = len(kfolded_indices[0])
    for i, k in enumerate(kfolded_indices):
        if len(k) < base_len:
            kfolded_indices[i] = np.pad(k, (0, base_len - len(k)), mode='edge')[None, ...]
        else:
            kfolded_indices[i] = k[None, ...]
    return np.concatenate(kfolded_indices, 0)


def _get_kfolded_indices_rgs(valid_indices, trainset, num_k, seed):
    np.random.seed(seed)
    valid_indices = np.array(valid_indices) # tarin idx?? need to check
    np.random.shuffle(valid_indices)
    if len(valid_indices) % num_k: # vaule > 0 will be true.
        valid_indices = np.pad(valid_indices, (0, num_k - len(valid_indices) % num_k), mode='edge')
    valid_indices = valid_indices.reshape(num_k, -1) 
    return valid_indices


class NbsDataset(VisionDataset):
    def __init__(self, dataset, group):
        self.dataset = dataset
        self.group = group

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        index = np.where(self.group == idx)[0][0]
        return img, label, index

    def __len__(self):
        return len(self.dataset)


class BaseDataLoader(object):
    def __init__(self, dataset, batch_size, cpus, with_index, seed, val_splitter):
        self.with_index = with_index
        self.dataset = self._get_dataset(dataset) # need to add custom dataset
        self.split_indices = val_splitter(self.dataset['train'], 0.1, seed)
        self.n_train = len(self.split_indices[0]) # length of the train idx
        self.n_val = len(self.split_indices[1]) # length of the validation idx
        self.n_test = len(self.dataset['test'])
        self.batch_size = batch_size
        self.cpus = cpus

    def load(self, phase):
        _f = {'train': lambda: self._train(),
              'val': lambda: self._val(),
              'test': lambda: self._test()}
        try:
            loader = _f[phase]()
            return loader
        except KeyError:
            raise ValueError('Dataset should be one of [train, val, test]')

    def _train(self):
        self.len = ceil(self.n_train / self.batch_size)
        sampler = SubsetRandomSampler(self.split_indices[0])
        dataset = NbsDataset(self.dataset['train'], self.groups) if self.with_index else self.dataset['train']
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _val(self):
        sampler = SubsetRandomSampler(self.split_indices[1])
        loader = DataLoader(self.dataset['train'], batch_size=self.batch_size,
                            sampler=sampler, num_workers=self.cpus,
                            pin_memory=True)
        return loader

    def _test(self):
        self.len = ceil(self.n_test / self.batch_size)
        loader = DataLoader(self.dataset['test'], batch_size=self.batch_size,
                            num_workers=self.cpus, pin_memory=True)
        return loader

    def _get_dataset(self, dataset):
        _d = {'cifar10': lambda: self._load_cifar10(), # dataset._load_cifar10() ? no specifed method.
              'cifar100': lambda: self._load_cifar100(),
              'mnist': lambda: self._load_mnist(),
              'svhn': lambda: self._load_svhn(),
              'svhn_extra': lambda: self._load_svhn(use_extra=True),
              'stl': lambda: self._load_stl(),
              'voc': lambda: self._load_voc(),
              'custom' : lambda: self._load_custom(),
              'custom2' : lambda: self._load_custom2(),
              'custom3' : lambda: self._load_custom3(),
              'sin' : lambda: self._load_sin(),
              'SP' : lambda: self._load_SP()}
        try:
            _dataset = _d[dataset]()
            return _dataset
        except KeyError:
            raise ValueError(
                "Dataset should be one of [mnist, cifar10"
                ", cifar100, svhn, svhn_extra, stl, voc]")
        
    def _load_custom(self):
        '''
        y =bx
        일반적인 linear regression을 가정하고 만든 데이터셋
        이미 noramlize되어 있으므로 따로 하지 않았다.
        '''
        n_train= 100000
        n_test = 2000
        # true beta
        beta = torch.ones([100,1])
        # mean 5, std 2
        train_X = torch.normal(0,1,size=(n_train,100))
        train_y = torch.mm(train_X,beta)
        test_X = torch.normal(0,1,size=(n_test,100))
        test_y = torch.mm(test_X,beta)
        trainset = CustomDataset(train_X,train_y)
        testset = CustomDataset(test_X,test_y)
        print(train_X.size())
        print(train_y.size())
        return {'train': trainset, 'test': testset}

    def _load_custom2(self):
        '''
        y = e(x) , e(x) ~ N(0,sqrt(x**2 + 1e-5))
        non-linear regresssion을 위한 데이터셋
        해본 결과 거의 mean만 예측하는 결과를 만들었다.
        '''
        n_train= 1000
        n_test = 200

        train_step =0.004
        test_step = 0.02
        start = -2
        end = -start

        train_X = torch.arange(start,end,train_step)
        train_sd = torch.sqrt(train_X**2 + 1e-05)
        train_y = torch.normal(torch.zeros(train_X.shape[0]),train_sd)

        test_X = torch.arange(start,end,test_step)
        test_sd = torch.sqrt(test_X**2 + 1e-05)
        test_y = torch.normal(torch.zeros(test_X.shape[0]),test_sd)

        train_X = train_X.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
        test_X = test_X.unsqueeze(1)
        test_y = test_y.unsqueeze(1)

        # y = torch.normal(torch.zeros(X.shape[0]),sd)
        # train_idx = torch.randint(0,100,(n_train,))
        # train_X = X[train_idx].unsqueeze(1)
        # train_y = y[train_idx].unsqueeze(1)

        # test_idx = torch.randint(0,100,(n_test,))
        # test_X = X[test_idx].unsqueeze(1)
        # test_y = y[test_idx].unsqueeze(1)
        
        # normalize
        train_X = (train_X - train_X.mean())/train_X.std()
        test_X = (test_X - test_X.mean())/test_X.std()

        trainset = CustomDataset(train_X,train_y)
        testset = CustomDataset(test_X,test_y)
        print(train_X.size())
        print(train_y.size())
        return {'train': trainset, 'test': testset}

    def _load_custom3(self):
        '''
        y = x + e(x) , e(x) ~ N(0,sqrt(x**2 + 1e-5))
        non-linear 이면서 y의 평균을 0이 아니게 만들어본 데이터셋
        '''
        n_train= 100000
        n_test = 2000

        train_step =0.004
        test_step = 0.02
        start = -2
        end = -start

        train_X = torch.normal(torch.tensor(0),torch.tensor(1),(n_train,))
        train_sd = torch.sqrt(train_X**2 + 1e-05)
        train_y = train_X + torch.normal(torch.zeros(train_X.shape[0]),train_sd)

        test_X = torch.normal(torch.tensor(0),torch.tensor(1),(n_test,))
        test_sd = torch.sqrt(test_X**2 + 1e-05)
        test_y = test_X + torch.normal(torch.zeros(test_X.shape[0]),test_sd)

        train_X = train_X.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
        test_X = test_X.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
        
        # normalize
        train_X = (train_X - train_X.mean())/train_X.std()
        test_X = (test_X - test_X.mean())/test_X.std()

        trainset = CustomDataset(train_X,train_y)
        testset = CustomDataset(test_X,test_y)
        print(train_X.size())
        print(train_y.size())
        return {'train': trainset, 'test': testset}
    
    def _load_sin(self):
        '''
        y = x + e(x) , e(x) ~ N(0,sqrt(x**2 + 1e-5))
        non-linear 이면서 y의 평균을 0이 아니게 만들어본 데이터셋
        '''
        n_train= 100000
        n_test = 2000

        train_X = torch.normal(0,2,(n_train,))   # (torch.rand((n_train,))-0.5)*5
        train_y = 3. + torch.sin(train_X) + torch.randn(n_train)*0.05
        test_X = torch.normal(0,2,(n_test,)) # (torch.rand((n_test,))-0.5)*5
        test_y = 3. + torch.sin(test_X) + torch.randn(n_test)*0.05

        train_X = train_X.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
        test_X = test_X.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
        
        # normalize
        train_X = (train_X - train_X.mean())/train_X.std()
        test_X = (test_X - test_X.mean())/test_X.std()

        trainset = CustomDataset(train_X,train_y)
        testset = CustomDataset(test_X,test_y)
        print(train_X.size())
        print(train_y.size())
        return {'train': trainset, 'test': testset}
    
    def _load_SP(self):
        '''
        2D-Gaussian process example
        '''
        n_train= 10000
        n_test = 200
        num_total = 100 # size*size
        # max_context = 50          
        # num_context = max_context  

        l1 = 0.6
        sigma= 1.0
        beta = torch.tensor([1.,1.])
        #-----------------------------------------------------------#
        # trainset 
        # generate random points
        # generate the obs values 

        train_mesh = torch.randn((n_train,num_total,2))*4 - 2
        train_X = torch.rand((n_train*num_total,2))

        # make gaussian kernel
        kernel = (sigma**2)*(torch.exp(-0.5*torch.cdist(train_mesh,train_mesh)/l1))
        kernel += ((2e-2)**2)*torch.eye(num_total)

        # make yval that follows the gaussian process.
        cholesky = torch.linalg.cholesky(kernel)
        train_y = (train_X @ beta).unsqueeze(1) + torch.bmm(cholesky,torch.normal(0,1,size=(n_train,num_total,1))).reshape(-1,1)
        
        #-----------------------------------------------------------#
        # testset 
        # generate random points
        test_mesh = torch.randn((n_test,num_total,2))*4 - 2
        test_X = torch.rand((n_test*num_total,2))

        # make gaussian kernel
        kernel = (sigma**2)*(torch.exp(-0.5*torch.cdist(test_mesh,test_mesh)/l1))
        kernel += ((2e-2)**2)*torch.eye(num_total)

        # make yval that follows the gaussian process.
        cholesky = torch.linalg.cholesky(kernel)
        test_y = (test_X @ beta).unsqueeze(1) + torch.bmm(cholesky,torch.normal(0,1,size=(n_test,num_total,1))).reshape(-1,1)
        
        # normalize
        train_X = (train_X - train_X.mean())/train_X.std()
        test_X = (test_X - test_X.mean())/test_X.std()
        
        trainset = CustomDataset(train_X,train_y)
        testset = CustomDataset(test_X,test_y)

        print(train_X.size())
        print(train_y.size())
        return {'train': trainset, 'test': testset}
    
    
    def _load_mnist(self):
        # trainset = Dataset(MNIST(root='.mnist', train=True, download=True,
                        #    transform=MnistNorm()), with_index=self.with_index)
        trainset = MNIST(root='.mnist', train=True, download=True,
                         transform=MnistNorm())
        testset = MNIST(root='.mnist', train=False, download=True,
                        transform=MnistNorm())
        return {'train': trainset, 'test': testset}

    def _load_cifar10(self):
        trainset = CIFAR10(root='.cifar10', train=True, download=True,
                           transform=get_transform(32, 4, 16)['train'])
        testset = CIFAR10(root='.cifar10', train=False, download=True,
                          transform=get_transform(32, 4, 16)['test'])
        return {'train': trainset, 'test': testset}

    def _load_cifar100(self):
        trainset = CIFAR100(root='.cifar100', train=True, download=True,
                           transform=get_transform(32, 4, 8)['train'])
        testset = CIFAR100(root='.cifar100', train=False, download=True,
                           transform=get_transform(32, 4, 8)['test'])
        return {'train': trainset, 'test': testset}

    def _load_svhn(self, use_extra=False):
        trainset = SVHN(root='.svhn', split='train', download=True,
                           transform=get_transform(32, 4, 20)['train'])
        testset = SVHN(root='.svhn', split='test', download=True,
                       transform=get_transform(32, 4, 20)['test'])
        if not use_extra:
            return {'train': trainset, 'test': testset}

        extraset = SVHN(root='.svhn', split='extra', download=True,
                           transform=get_transform(32, 4, 20)['train'])
        return {'train': trainset + extraset, 'test': testset}

    def _load_stl(self):
        trainset = STL10(root='.stl', split='train', download=True,
                           transform=get_transform(96, 12, 32, 'stl')['train'])
        testset = STL10(root='.stl', split='test', download=True,
                        transform=get_transform(96, 12, 32, 'stl')['test'])
        return {'train': trainset, 'test': testset}

    def _load_voc(self):
        trans = get_transform(513, 0, 0, 'voc')
        trainset = VOCSegmentation(root='.voc', image_set='train', download=True,
                                transforms=trans['train'])
        testset = VOCSegmentation(root='.voc', image_set='val', download=True,
                                transforms=trans['test'])
        return {'train': trainset, 'test': testset}


class GeneralDataLoaderCls(BaseDataLoader):
    def __init__(self, dataset, batch_size, cpus,
                 seed=0, val_splitter=_get_split_indices_cls):
        super().__init__(dataset, batch_size, cpus, False, seed, val_splitter)


class NbsDataLoaderCls(BaseDataLoader):
    def __init__(self, dataset, batch_size, n_a, cpus,
                 seed=0, val_splitter=_get_split_indices_cls):
        super().__init__(dataset, batch_size, cpus, True, seed, val_splitter) # done. define the train,valid,test set
        self.n_a = n_a
        self.groups = _get_kfolded_indices_rgs(self.split_indices[0],
                                               self.dataset['train'],
                                               n_a, seed) # (n_a*-1)
 

class GeneralDataLoaderRgs(BaseDataLoader):
    def __init__(self, dataset, batch_size, cpus,
                 seed=0, val_splitter=_get_split_indices_rgs):
        super().__init__(dataset, batch_size, cpus, False, seed, val_splitter)


class NbsDataLoaderRgs(BaseDataLoader):
    def __init__(self, dataset, batch_size, n_a, cpus,
                 seed=0, val_splitter=_get_split_indices_rgs):
        super().__init__(dataset, batch_size, cpus, True, seed, val_splitter)
        self.n_a = n_a
        self.groups = _get_kfolded_indices_rgs(self.split_indices[0], # self.split_indices[0]: trainset, [1] :val-set
                                               self.dataset['train'],
                                               n_a, seed)


class GeneralDataLoaderSeg(GeneralDataLoaderRgs):
    pass


class NbsDataLoaderSeg(NbsDataLoaderRgs):
    pass
