import torch
from torch.nn.functional import one_hot
from torch.distributions.exponential import Exponential
from scipy.special import softmax

import h5py
import numpy as np
from tqdm import tqdm
from time import time
from pathlib import Path

from utils.metrics import calc_ece, calc_nll_brier, calc_nll_brier_mc
from runners.base_runner import gather_tensor
from runners.cnn_runner import CnnRunner


class NbsRunner(CnnRunner):
    def __init__(self, loader, model, optim, lr_scheduler, num_epoch,
                 loss_with_weight, val_metric, test_metric, logger,
                 model_path, rank, epoch_th, num_mc, adv_training):
        self.num_mc = num_mc
        self.n_a = loader.n_a
        self.epoch_th = epoch_th
        self.alpha = torch.ones([1, self.n_a])
        super().__init__(loader, model, optim, lr_scheduler, num_epoch, loss_with_weight,
                         val_metric, test_metric, logger, model_path, rank, adv_training)
        self.save_kwargs['alpha'] = self.alpha # saving in the dictionary
        self._update_weight() # sampling the variables following dirichlet dist. 

    def _update_weight(self):
        if self.epoch > self.epoch_th:
            self.alpha = Exponential(torch.ones([1, self.n_a])).sample() # sampling the variables following dirichlet dist. 

    def _calc_loss(self, img, label, idx): # theses options in batch
        n0 = img.size(0)
        w = self.alpha[0, idx].cuda()

        output = self.model(img.cuda(non_blocking=True),
                            self.alpha.repeat_interleave(n0, 0))
        for _ in range(output.dim() - w.dim()):
            w.unsqueeze_(-1)
        label = label.cuda(non_blocking=True)
        loss_ = 0
        for loss, _w in self.loss_with_weight:
            _loss = _w * loss(output, label, w)
            loss_ += _loss
        return loss_

    @torch.no_grad()
    def _valid_a_batch(self, img, label, with_output=False):
        self._update_weight()
        self.model.eval()
        output = self.model(img.cuda(non_blocking=True), self.num_mc) # output -> (num_mc,batch_size,1) mc : motecarlo sampling ?
        label = label.cuda(non_blocking=True)
        result = self.val_metric(output.mean(0), label) 
        if with_output:
            result = [result, output]
        return result

    def test(self, is_seg):
        self.load('model.pth')
        loader = self.loader.load('test')
        if self.rank == 0:
            t_iter = tqdm(loader, total=self.loader.len)
        else:
            t_iter = loader

        outputs = []
        labels = []
        metrics = []
        self.model.eval()
        for img, label in t_iter:
            _metric, output = self._valid_a_batch(img, label, with_output=True)
            labels += [gather_tensor(label).cpu().numpy()]
            outputs += [gather_tensor(output).cpu().numpy()]
            metrics += [gather_tensor(_metric).cpu().numpy()]
        if is_seg:
            met = np.concatenate(metrics).mean()
            self.log(f"[Test] MeanIOU: {met:.2f}", 'info')
            save_path = Path(self.model_path) / 'infer'
            save_path.mkdir(parents=True, exist_ok=True)
            index = 0
            for out, label in zip(outputs, labels):
                for i in range(label.shape[0]):
                    l = label[i]
                    o = out[:, i]

                    with h5py.File(f"{save_path}/{index}.h5", 'w') as h:
                        h.create_dataset('output', data=o)
                        h.create_dataset('label', data=l)
                    index += 1
        else:
            labels = np.concatenate(labels)
            outputs = np.concatenate(outputs, axis=1)
            acc = (outputs.mean(0).argmax(-1) == labels).mean() * 100
            ece = calc_ece(softmax(outputs, -1).mean(0), labels)
            # nll, brier = calc_nll_brier_mc(outputs, labels)
            # print("Becarful, the num_class set as 1")
            log = f"[Test] ACC: {acc:.2f}, ECE : {ece:.2f}, "
            # log += f"NLL : {nll:.2f}, Brier : {brier:.2f}"
            self.log(log, 'info')
            with h5py.File(f"{self.model_path}/output.h5", 'w') as h:
                h.create_dataset('output', data=outputs)
                h.create_dataset('label', data=labels)
