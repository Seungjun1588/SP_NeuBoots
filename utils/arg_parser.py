import os
import yaml
import copy
import logging
from pathlib import Path

import torch
from torch.nn import *
from torch.optim import *
import torch.distributed as dist
from torch.optim.lr_scheduler import *
from torch.nn.parallel import DistributedDataParallel

from utils.metrics import *
from models import _get_model


torch.backends.cudnn.benchmark = True


class Argments(object):
    @staticmethod
    def _file_load(yaml_file):
        # f-string ; r means raw string
        with open(fr'{yaml_file}') as f:
            y = yaml.safe_load(f)
        return y

    @staticmethod
    def _module_load(d, part, **kargs):
        module_obj = eval(d[part]['name'])
        module_args = copy.deepcopy(d[part])
        module_args.update(kargs)
        del module_args['name']
        part = module_obj(**module_args)
        return part

    def _modules_load(self):
        for k, v in self._y.items():
            if 'module' in k:
                setattr(self, k, dict())
                module = self.__dict__[k]
                module['model'] = _get_model(**v['model'], model_type=self['setup/model_type']).cuda()
                if self['setup/phase'] != 'infer': # mean train?
                    module['optim'] = self._module_load(v, part='optim',
                                                        params=module['model'].parameters())

                    module['lr_scheduler'] = self._module_load(v, part='lr_scheduler',
                                                               optimizer=module['optim'])
                    # loss = [eval(l)(**v['loss_args'][l]) for l in v['loss']]
                    loss = [eval(v['loss'])(**v['loss_args'])]
                    # print(loss) # [CrossEntropyLoss()]
                    # print(module['loss_with_weight']) # [(CrossEntropyLoss(), 1.0)]
                    module['loss_with_weight'] = list(zip(loss,v['loss_weight']))   # list(zip(loss, v['loss_weight']))
                    module['val_metric'] = eval(v['val_metric'])(**v['metric_args'])  
                    module['test_metric'] = eval(v['test_metric'])(**v['metric_args'])
                else:
                    pass

    def __init__(self, yaml_file, cmd_args):
        # load the yaml file
        self.file_name = yaml_file
        self._y = self._file_load(yaml_file)
        
        # set the directory with written status
        if cmd_args.gpus != "-1":
            self['setup/gpus'] = cmd_args.gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = self["setup/gpus"]

        # adding the args
        self['setup/index'] = cmd_args.index
        self['setup/phase'] = cmd_args.phase
        self['setup/local_rank'] = cmd_args.local_rank

        # the number of the possible gpus
        world_size = len(self["setup/gpus"].replace(',', "").replace("'", ""))

        # specified in loaded yaml file(type,name and dataset)
        model_path = f"outs/{self['setup/model_type']}/{self['module/model/name']}"
        model_path += f"/{self['path/dataset']}"
        if self['setup/index'] != -1:
            model_path += f"_{self['setup/index']}"
        if self['path/postfix'] != 'none':
            model_path += f"_{self['path/postfix']}"
        self['path/model_path'] = model_path
        # making the folder
        Path(model_path).mkdir(parents=True, exist_ok=True)

        # setting the python env
        print("setting the python env")
        print("cmd_args.local_rank: ",cmd_args.local_rank)
        torch.cuda.set_device(cmd_args.local_rank)
        # print("setting the python env done. ")
        # print("init_method : ",f'file://{Path(model_path).resolve()}/sharedfile')
        # print("world_size :  ",world_size)
        # print("rank : ",self["setup/local_rank"])
        # torch.distributed.init_process_group(backend= "gloo",
        #                                      init_method=f'file://{Path(model_path).resolve()}/sharedfile',
        #                                      world_size=world_size,
        #                                      rank=self['setup/local_rank'])
        self['setup/rank'] = self['setup/local_rank']
        self['setup/dist_size'] = 1 # dist.get_world_size()

        self._modules_load()

    def reset(self):
        for k, v in list(self.__dict__.items()):
            if 'module' in k:
                del self.__dict__[k]
        torch.cuda.empty_cache()
        self._modules_load()

    def _get(self, *keys):
        v = self._y
        for k in keys:
            v = v[k]
        return v

    def _update(self, *keys, value):
        k = self._y
        for i in range(len(keys) - 1):
            k.setdefault(keys[i], {})
            k = k[keys[i]]
        k[keys[-1]] = value

    def __str__(self):
        return f'{self.file_name}\n{self._y}'

    def __contains__(self, item):
        def search_recursively(d, t):
            for k, v in d.items():
                if k == t:
                    return True
                elif isinstance(v, dict):
                    search_recursively(v, t)
            return False

        return search_recursively(self._y, item)

    def __getitem__(self, key):
        return self._get(*key.split('/'))

    def __setitem__(self, key, value):
        self._update(*key.split('/'), value=value)


if __name__ == '__main__':
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.INFO)
    log.addHandler(stream_handler)
    log.addHandler(file_handler)

    Args = Argments('test.yaml')
    Args._update('path', 'abcd', 'efgh', value='zzzz')
    Args['path/cccc/dddd'] = 'ffff'
    log.debug(Args)
    log.debug(Args['path/cccc/dddd'])
    
    # print(Args)
    # print('path' in Args)
    # print(Args['path/abcd/efgh'])
    # print(Args['path/cccc/dddd'])
    # print(Args.module['lr_scheduler'])
