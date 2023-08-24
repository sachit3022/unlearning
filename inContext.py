# implementation of simple exampple to show transformers exibit incontext learning.
"""
This module is used to run experiments on the unlearning project.
Returns:
    experiments: experiments folder.
"""

import copy
import torch
import torch.nn.functional as F
from torch import nn
import logging
import config
import network
import trainer as tr
from trainer import Trainer,  TrainerSettings, count_parameters
from dataset import create_injection_dataloaders,create_dataloaders_missing_class,create_dataloaders_uniform_sampling,get_finetune_dataloaders,create_celeba_dataloaders,create_celeba_id_dataloaders
from score import compute_unlearning_metrics,compute_retrain_unlearning_metrics, compute_acc_metrics # 2 types of unleaning metrics



def main(args):
    

    # set loggers
    logger = logging.getLogger()
    logger.info("config: {}".format(args))


    # load model
    net = getattr(network, args.model.name)
    net = net(**args.model.model_args)
    if args.model.checkpoint is not None:
        net.load_state_dict(torch.load(
            args.model.checkpoint, map_location=args.DEVICE))
    net.name = args.model.name
    logger.info(f"Model has {count_parameters(net)} parameters")    


    dataloader_fn = create_dataloaders_uniform_sampling #create_dataloaders_missing_class #create_celeba_dataloaders #create_celeba_id_dataloaders #create_dataloaders_uniform_sampling

    
    # scratch trainer for perfect baseline
    optimizer_config = getattr(tr, args.trainer.optimizer.type + "OptimConfig" )(** args.trainer.optimizer)
    scheduler_config = getattr(tr, args.trainer.scheduler.type + "SchedulerConfig" )(**args.trainer.scheduler)
    
    dataloaders =dataloader_fn(config=args)
    trainer_settings = TrainerSettings(name = net.name,optimizer=optimizer_config, scheduler=scheduler_config, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.trainer.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    trainer = Trainer(model=copy.deepcopy(net),dataloaders=dataloaders,trainer_args=trainer_settings)
    if args.trainer_checkpoint is not None:
        trainer = trainer.load_from_checkpoint(args.trainer_checkpoint)    
    if args.trainer.train:
        trainer.train(epochs=args.trainer.epochs)    
    
    return 


if __name__ == "__main__":

    args = config.set_config()
    main(args)


