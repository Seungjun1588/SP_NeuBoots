import argparse

from utils.logger import get_logger
from utils.arg_parser import Argments
from loader.data_loader import GeneralDataLoaderCls, NbsDataLoaderCls,\
                               GeneralDataLoaderSeg, NbsDataLoaderSeg,NbsDataLoaderRgs
from runners.cnn_runner import CnnRunner
from runners.nbs_runner import NbsRunner
from runners.mcd_runner import McdRunner


def main():
    # making instance
    argparser = argparse.ArgumentParser()
    # adding the values
    argparser.add_argument("yaml")
    argparser.add_argument("--phase", default="train", type=str) # train / infer 
    argparser.add_argument("--index", default=-1, type=int) # meaning ? -> used in file name(spcific version : file1,file2,...) -1 means nothing is added in file name.
    argparser.add_argument("--gpus", default="0", type=str)
    argparser.add_argument("--local_rank", default=0, type=int) # meaning ? logger level?
    # save 
    cmd_args = argparser.parse_args(["example"])

    ###################### <- here
    # local_rank : specified in yaml file
    # rank : same as local_rank(from dist ~)

    arg = Argments(f"scripts/{cmd_args.yaml}.yaml", cmd_args) # stll many funcs to read
    setup = arg['setup']
    model_path = arg['path/model_path']
    logger = get_logger(f"{model_path}/log.txt")

    if setup['rank'] == 0:
        logger.info(arg)

    model_type = setup['model_type']
    dataset = arg['path/dataset']
    is_seg = False

    # loading the data loader and runner. (It depends on the model type.)
    if 'nbs' in model_type: # added in n_a (comparing the else loop)
        if 'seg' in model_type: # segment task
            is_seg = True
            _data_loader = NbsDataLoaderSeg # n_a exists
        else: # this part is the core of the paper
            _data_loader = NbsDataLoaderRgs # n_a exists
        data_loader = _data_loader(dataset, setup['batch_size'],
                                    setup['n_a'], setup['cpus'], setup['seed'])
        runner = NbsRunner(data_loader, **arg.module, num_epoch=setup['num_epoch'], # core part 
                            logger=logger, model_path=model_path, rank=setup['rank'],
                            epoch_th=setup['epoch_th'], num_mc=setup['num_mc'],
                            adv_training=setup['adv_training'])
    else:
        if 'seg' in model_type:
            is_seg = True
            _data_loader = GeneralDataLoaderSeg # nothing
        else:
            _data_loader = GeneralDataLoaderCls # no n_a
        data_loader = _data_loader(dataset, setup['batch_size'],
                                   setup['cpus'], setup['seed'])
        if 'mcd' in model_type: # mcd ?? num_mc ? 
            runner = McdRunner(data_loader, **arg.module, num_epoch=setup['num_epoch'],
                               logger=logger, model_path=model_path, rank=setup['rank'],
                               num_mc=setup['num_mc'], adv_training=setup['adv_training'])
        else:
            runner = CnnRunner(data_loader, **arg.module, num_epoch=setup['num_epoch'],
                            logger=logger, model_path=model_path, rank=setup['rank'],
                            adv_training=setup['adv_training'])

    if setup['phase'] == 'train':
        runner.train() 
        runner.test(is_seg) 
    elif setup['phase'] == 'test':
        runner.test(is_seg)


if __name__ == "__main__":
    main()
