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
from trainer import Trainer, AdverserialTrainer, NNTrainer, TrainerSettings,BoundaryUnlearning, RLFTrainer, count_parameters
from datasets import get_finetune_dataloaders, get_scratch_datatloaders,TrainerDataLoaders,get_celeba_datasets,get_celeba_dataloaders,get_cifar10_dataloaders,get_cifar10_datasets
from evaluation import SVC_MIA
from network import MTLLoss
import argparse
import time

"""Launch to see if everything works."""
from unl_algorithms.pruning import unlearn
from unl_algorithms import surgery_unlearning

def pruning_unleanring(model,seed,device,dataset="celeba"):
    unlearn_model = unlearn(
        pretrained_model_or_path=model,
        seed=seed,
        device=device,
        dataset=dataset
    )
    torch.save(unlearn_model.state_dict(), f"models/pruning_unlearning_{seed}.pth")
    
def boundary_unlearning(args, model, dataloaders):

    optimizer_config = getattr(tr, args.finetune.optimizer.type + "OptimConfig" )(** args.finetune.optimizer)
    scheduler_config = getattr(tr, args.finetune.scheduler.type + "SchedulerConfig" )(**args.finetune.scheduler)
    nn_trainer_settings = TrainerSettings(name = f"BU_{args.SEED}_{model.name}",optimizer=optimizer_config, scheduler=scheduler_config, num_classes=args.data.num_classes,log_path= args.directory.LOG_PATH,device=args.device,loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1), **{k:v for k,v in args.finetune.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    nn_trainer = BoundaryUnlearning(model=model,dataloaders=dataloaders,trainer_args=nn_trainer_settings)
    nn_trainer.train(args.finetune.epochs)
    return nn_trainer.model

def finetune_unlearning(args, model, dataloaders):
    #how forgetting happens in the model when finetuining.
    optimizer_config = getattr(tr, args.finetune.optimizer.type + "OptimConfig" )(**args.finetune.optimizer)
    scheduler_config = getattr(tr, args.finetune.scheduler.type + "SchedulerConfig" )(**args.finetune.scheduler)
    trainer_settings = TrainerSettings(name = f"finetune_{args.SEED}_{model.name}__{args.data.num_classes}",optimizer=optimizer_config, scheduler=scheduler_config, num_classes=args.data.num_classes, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.finetune.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    finetune_dataloaders = get_finetune_dataloaders(dataloaders)
    finetune_trainer = Trainer(model=model,dataloaders=finetune_dataloaders,trainer_args=trainer_settings)
    finetune_trainer.train(args.finetune.epochs)
    return finetune_trainer.model

def scrubs_unlearning(args,model,dataloaders):
    optimizer_config = getattr(tr, args.finetune.optimizer.type + "OptimConfig" )(**args.finetune.optimizer)
    scheduler_config = getattr(tr, args.finetune.scheduler.type + "SchedulerConfig" )(**args.finetune.scheduler)
    trainer_settings = TrainerSettings(name = f"scrubs_{args.SEED}_{model.name}__{args.data.num_classes}",optimizer=optimizer_config, scheduler=scheduler_config,num_classes=args.data.num_classes, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.finetune.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    finetune_trainer = AdverserialTrainer(model=model,dataloaders=dataloaders,trainer_args=trainer_settings)
    finetune_trainer.train(args.finetune.epochs)
    return finetune_trainer.model


def nearest_neighbor_unlearning(args, model, dataloaders):

    optimizer_config = getattr(tr, args.finetune.optimizer.type + "OptimConfig" )(** args.finetune.optimizer)
    scheduler_config = getattr(tr, args.finetune.scheduler.type + "SchedulerConfig" )(**args.finetune.scheduler)
    nn_trainer_settings = TrainerSettings(name = f"NN_{args.SEED}_{model.name}__{args.data.num_classes}",optimizer=optimizer_config, scheduler=scheduler_config, num_classes=args.data.num_classes,log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.finetune.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    nn_trainer = NNTrainer(model=model,dataloaders=dataloaders,trainer_args=nn_trainer_settings)
    nn_trainer.train(args.finetune.epochs)
    return nn_trainer.model

def random_label_fliping(args, model, dataloaders):

    optimizer_config = getattr(tr, args.finetune.optimizer.type + "OptimConfig" )(** args.finetune.optimizer)
    scheduler_config = getattr(tr, args.finetune.scheduler.type + "SchedulerConfig" )(**args.finetune.scheduler)
    nn_trainer_settings = TrainerSettings(name = f"RLF_{args.SEED}_{model.name}",optimizer=optimizer_config, scheduler=scheduler_config, num_classes=args.data.num_classes,log_path= args.directory.LOG_PATH,device=args.device,loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1), **{k:v for k,v in args.finetune.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    nn_trainer = RLFTrainer(model=model,dataloaders=dataloaders,trainer_args=nn_trainer_settings)
    nn_trainer.train(args.finetune.epochs)
    return nn_trainer.model

##################################     New unlearnig algorithm goes here       ######################################
def train(args,net,dataloaders):
    
    optimizer_config = getattr(tr, args.trainer.optimizer.type + "OptimConfig" )(** args.trainer.optimizer)
    scheduler_config = getattr(tr, args.trainer.scheduler.type + "SchedulerConfig" )(**args.trainer.scheduler)

    trainer_settings = TrainerSettings(name =f"train_{args.SEED}_{args.experiment}" ,optimizer=optimizer_config, scheduler=scheduler_config,num_classes=args.data.num_classes, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.trainer.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    trainer = Trainer(model=copy.deepcopy(net),dataloaders=dataloaders,trainer_args=trainer_settings)
    if args.trainer.checkpoint is not None:
        trainer = trainer.load_from_checkpoint(args.trainer.checkpoint)
    trainer.test_epoch()
    if args.trainer.train:
        trainer.train(epochs=args.trainer.epochs)
    return trainer.model

def training_from_scratch(args, net, dataloaders):
    
    optimizer_config = getattr(tr, args.scratch_trainer.optimizer.type + "OptimConfig" )(** args.scratch_trainer.optimizer)
    scheduler_config = getattr(tr, args.scratch_trainer.scheduler.type + "SchedulerConfig" )(**args.scratch_trainer.scheduler)

    scratch_trainer_settings = TrainerSettings(name = f"scratch_{args.SEED}_{net.name}_{args.data.num_classes}",optimizer=optimizer_config, scheduler=scheduler_config,num_classes=args.data.num_classes, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.scratch_trainer.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    scratch_data_loaders = TrainerDataLoaders(**{"train":dataloaders.retain,"retain":dataloaders.retain,"forget":dataloaders.forget,"val":dataloaders.val,"test":dataloaders.test})
    scratch_trainer = Trainer(model=copy.deepcopy(net),dataloaders=scratch_data_loaders,trainer_args=scratch_trainer_settings)
    
    if args.scratch_trainer.checkpoint is not None:
       scratch_trainer = scratch_trainer.load_from_checkpoint(args.scratch_trainer.checkpoint)
    if args.scratch_trainer.train:
       scratch_trainer.train(epochs=args.scratch_trainer.epochs)

    return scratch_trainer.model


#####################################################################################################################


def main(args):
    

    # set loggers
    logger = logging.getLogger()
    logger.info(f"config: {args}")

    # load model
    net = getattr(network, args.model.name)
    net = net(**args.model.model_args)
    if args.model.checkpoint is not None:
        net.load_state_dict(torch.load(
            args.model.checkpoint, map_location=args.DEVICE))
    net.name = args.model.name
    logger.info(f"Model has {count_parameters(net)} parameters")    

    start_time = time.time()
    train_loader,retain_loader, forget_loader, validation_loader,test_loader = get_celeba_dataloaders(args,balanced=False) #get_celeba_dataloaders
    dataloaders = TrainerDataLoaders(**{"train":train_loader,"retain":retain_loader,"forget":forget_loader,"val":validation_loader,"test":test_loader})
    logger.info(f"Data loaded in {time.time()-start_time} seconds")

    #scratch_model = training_from_scratch(args, net, dataloaders)
    #full_model = train(args, net, dataloaders)
    unleart_model = nearest_neighbor_unlearning(args,net,dataloaders) #finetune_unlearning

    #unleart_model = surgery_unlearning(net,args.SEED,args.device,{"retain":retain_loader.dataset,"forget":forget_loader.dataset}) #boundary_unlearning(args,net,dataloaders)
    #torch.save(unleart_model.state_dict(), f"models/surgery_celeba_unlearning_{args.SEED}.pth")
  

    
    #pruning_unleanring(net,args.SEED,args.device)

    #evaluation
    #SVC_MIA(shadow_train = retain_loader,shadow_test = test_loader, target_train = None, target_test = forget_loader, model=unleart_model,device=args.device)
    
    return 




if __name__ == "__main__":
   
    
    parser = argparse.ArgumentParser(description='Unlearning')
    args = config.set_config(parser)    
    main(args)
    
    

