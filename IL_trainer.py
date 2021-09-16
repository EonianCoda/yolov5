#built-in

from IL_utils.IL_cfg import IL_cfg
import os
import numpy as np
# torch
import torch
from torch import optim




#yolov5 package
from models.yolo import Model

class IL_Trainer(object):
    def __init__(self, cfg, nc:int):
        """
        Args:
            cfg: parser
            nc: number classes
        """
        hyp = cfg.hpy
        self.il_cfg = IL_cfg(cfg)
        self.cfg = cfg

        # create model
        self.model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).cuda()
        self.cur_state = self.il_cfg['start_state']
        

    def next_state(self):
        if self.cur_state + 1 >= self.il_cfg['end_state']:
            raise ("Next state dose not exist!")
        self.cur_state += 1
        num_new_classes = self.il_cfg.states.states[self.cur_state]['num_new_classes']
        # expand_classes
        self.model.expand_classes(num_new_classes)
        # create optimizer
    #     if config.TRAIN_OPTIMIZER.lower() == 'adam':
    #         self.optimizer = optim.Adam(params=self.model.parameters(),
    #                                 lr=config.learning_rate / config.batch,
    #                                 betas=(0.9, 0.999),
    #                                 eps=1e-08,
    #                                 )
    #     elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
    #         self.optimizer = optim.SGD(params=self.model.parameters(),
    #                                 lr=config.learning_rate / config.batch,
    #                                 momentum=config.momentum,
    #                                 weight_decay=config.decay,
    #                                 )
        
    #     # create scheduler
    #     def burnin_schedule(i):
    #         """learning rate setup
    #         """
    #         if i < config.burn_in:
    #             factor = pow(i / config.burn_in, 4)
    #         elif i < config.steps[0]:
    #             factor = 1.0
    #         elif i < config.steps[1]:
    #             factor = 0.1
    #         else:
    #             factor = 0.01
    #         return factor
    #     self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, burnin_schedule)

    #     self.criterion = Yolo_loss(device=device, batch=self.config.batch // self.config.subdivisions, n_classes=self.config.classes)

    #     self.init_pretrained()
    #     self.init_training_dataset()

    # def init_training_dataset(self):
    #     """initialize the training dataset
    #     """
    #     self.train_dataset = Yolo_dataset(self.config.train_label, self.config, train=True)
    #     self.train_dataloader = DataLoader(self.train_dataset, 
    #                                         batch_size=self.config.batch // self.config.subdivisions, 
    #                                         shuffle=True,
    #                                         num_workers=8, 
    #                                         pin_memory=True, 
    #                                         drop_last=True,
    #                                         collate_fn=collate)

    # def init_pretrained(self):
    #     """load the checkpoint, resume training
    #     """
    #     # read checkpoint
    #     if self.config.pretrained:
    #         ckp = torch.load(self.config.pretrained)
    #         # only use conv137
    #         if not ckp.get('model_state_dict'):
    #             model_weight = ckp
    #             self.model.init_conv137(model_weight)
    #         # read all tools
    #         else:
    #             self.model.load_state_dict(ckp['model_state_dict'])
    #             self.optimizer.load_state_dict(ckp['optimizer_state_dict'])
    #             self.scheduler.load_state_dict(ckp['scheduler_state_dict'])

    # def save_ckp(self, epoch:int):
    #     """save the checkpoint
    #     """
    #     save_prefix = 'Yolov4_epoch'
    #     save_path = os.path.join(self.config.checkpoints, f'{save_prefix}{epoch}.pth')
    #     data = {'model_state_dict': self.model.state_dict(),
    #             'optimizer_state_dict': self.optimizer.state_dict(),
    #             'scheduler_state_dict': self.scheduler.state_dict()}
    #     torch.save(data, save_path)
        

    # def auto_delete(self, epoch:int):
    #     save_prefix = 'Yolov4_epoch'
    #     if epoch % 5 == 0:
    #         for i in range(1, epoch + 1):
    #             if i % 5 == 0:
    #                 continue
    #             save_path = os.path.join(self.config.checkpoints, f'{save_prefix}{i}.pth')
    #             if os.path.isfile(save_path):
    #                 os.remove(save_path)

