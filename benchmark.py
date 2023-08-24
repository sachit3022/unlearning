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
import json
import network
import trainer as tr
from trainer import Trainer, AdverserialTrainer, NNTrainer, TrainerSettings, count_parameters
from dataset import create_injection_dataloaders,create_dataloaders_missing_class,create_dataloaders_uniform_sampling,get_finetune_dataloaders,create_celeba_dataloaders,create_celeba_id_dataloaders
from score import compute_unlearning_metrics,compute_retrain_unlearning_metrics, compute_acc_metrics # 2 types of unleaning metrics
from network import MTLLoss

def finetune_unlearning(args, model, dataloaders):
    #how forgetting happens in the model when finetuining.
    optimizer_config = getattr(tr, args.finetune.optimizer.type + "OptimConfig" )(**args.finetune.optimizer)
    scheduler_config = getattr(tr, args.finetune.scheduler.type + "SchedulerConfig" )(**args.finetune.scheduler)
    trainer_settings = TrainerSettings(name = f"finetune_{model.name}",optimizer=optimizer_config, scheduler=scheduler_config, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.finetune.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    finetune_dataloaders = get_finetune_dataloaders(dataloaders)
    finetune_trainer = Trainer(model=model,dataloaders=finetune_dataloaders,trainer_args=trainer_settings)
    finetune_trainer.train(args.finetune.epochs)
    return finetune_trainer.model

def scrubs_unlearning(args,model,dataloaders):
    optimizer_config = getattr(tr, args.finetune.optimizer.type + "OptimConfig" )(**args.finetune.optimizer)
    scheduler_config = getattr(tr, args.finetune.scheduler.type + "SchedulerConfig" )(**args.finetune.scheduler)
    trainer_settings = TrainerSettings(name = f"scrubs_{model.name}",optimizer=optimizer_config, scheduler=scheduler_config, log_path= args.disjctory.LOG_PATH,device=args.device, **{k:v for k,v in args.finetune.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    finetune_trainer = AdverserialTrainer(model=model,dataloaders=dataloaders,trainer_args=trainer_settings)
    finetune_trainer.train(args.finetune.epochs)
    return finetune_trainer.model


def nearest_neighbor_unlearning(args, model, dataloaders):

    optimizer_config = getattr(tr, args.finetune.optimizer.type + "OptimConfig" )(** args.finetune.optimizer)
    scheduler_config = getattr(tr, args.finetune.scheduler.type + "SchedulerConfig" )(**args.finetune.scheduler)
    nn_trainer_settings = TrainerSettings(name = f"NN_{model.name}",optimizer=optimizer_config, scheduler=scheduler_config, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.finetune.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    nn_trainer = NNTrainer(model=model,dataloaders=dataloaders,trainer_args=nn_trainer_settings)
    nn_trainer.train(args.finetune.epochs)
    return nn_trainer.model

##################################     New unlearnig algorithm goes here       ######################################



#####################################################################################################################


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

    
    ##### CONFIGURE TRAINING #####
    # type of create_dataloaders_missing_class, create_dataloaders_uniform_sampling, create_injection_dataloaders
    # type of trainer: Trainer, AdverserialTrainer, NNTrainer
    #if you want custom trainer, create a new class in trainer.py and import it here and use it
    # type of compute_unlearning_metrics: compute_unlearning_metrics, compute_retrain_unlearning_metrics
    #compute_unlearning_metrics: test and forget set discriminators used if forget and test are from the same distribution
    #compute_retrain_unlearning_metrics: retrain from scratch model and unlearnt model outputs as  discriminators used if forget and test are from different distributions
    ##########################
    
    dataloader_fn = create_celeba_dataloaders #create_dataloaders_missing_class #create_celeba_dataloaders #create_celeba_id_dataloaders #create_dataloaders_uniform_sampling

    
    # scratch trainer for perfect baseline
    optimizer_config = getattr(tr, args.trainer.optimizer.type + "OptimConfig" )(** args.trainer.optimizer)
    scheduler_config = getattr(tr, args.trainer.scheduler.type + "SchedulerConfig" )(**args.trainer.scheduler)
    
    """
    scratch_trainer_settings = TrainerSettings(name = "scratch_"+net.name,optimizer=optimizer_config, scheduler=scheduler_config, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.trainer.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    scratch_data_loaders = dataloader_fn(config=args,scratch=False)
    scratch_trainer = Trainer(model=copy.deepcopy(net),dataloaders=scratch_data_loaders,trainer_args=scratch_trainer_settings)
    
    if args.scratch_trainer_checkpoint is not None:
       scratch_trainer = scratch_trainer.load_from_checkpoint(args.scratch_trainer_checkpoint)
    if args.trainer.train:
       scratch_trainer.train(epochs=args.trainer.epochs)
    """
    #mtl_loss_fn = MTLLoss(heads=range(40)) #range(40)[8,15,19]
    #loss_fn=mtl_loss_fn

    # train model on full data
    dataloaders =dataloader_fn(config=args)
    trainer_settings = TrainerSettings(name = net.name,optimizer=optimizer_config, scheduler=scheduler_config, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.trainer.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    trainer = Trainer(model=copy.deepcopy(net),dataloaders=dataloaders,trainer_args=trainer_settings)
    if args.trainer_checkpoint is not None:
        trainer = trainer.load_from_checkpoint(args.trainer_checkpoint)
    trainer.test_epoch()
    
    if args.trainer.train:
        trainer.train(epochs=args.trainer.epochs)

    # compute memorization score



    """
    mia_scores = compute_retrain_unlearning_metrics(args, trainer.model, scratch_trainer.model, dataloaders) # or can use compute_unlearning_metrics
    scores = compute_acc_metrics(args, trainer.model,scratch_trainer.model,dataloaders)
    logger.info(f"scores before unlearning for is {scores}")
    

    #insert unlearning algorithm here.
    for unlearning_funcs in [scrubs_unlearning,nearest_neighbor_unlearning,finetune_unlearning]:
        #parallelise this step.
        model = copy.deepcopy(trainer.model)
        unleart_model = unlearning_funcs(args,model,dataloaders)
        mia_scores = compute_retrain_unlearning_metrics(args, unleart_model, scratch_trainer.model, dataloaders)
        logger.info(f"MIA score after unlearning for is {mia_scores}")
        scores = compute_acc_metrics(args, unleart_model,scratch_trainer.model,dataloaders)
        logger.info(f"scores after unlearning for is {scores}")
    """
    
    
    return 


if __name__ == "__main__":

    args = config.set_config()
    main(args)

    # Comments:
    # not much progress while training the model on the forget set.
    # why is there no change in gradients when there is change in the loss? investigate the mia_dataset and comeup with solution.
    # no matter what the train and test are the same
    # by using label smoothing the mia score decreases and test and train remians the same.
    # best to test on faces dataset. vggface2
    # think of ideas to debug the gradient problem.


