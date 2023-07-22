"""
This module is used to run experiments on the unlearning project.
Returns:
    experiments: experiments folder.
"""
import copy
import torch
from torch import nn
import logging
import config
import network
from trainer import Trainer, TrainerSettings, OptimizerConfig, SchedulerConfig, count_parameters, get_optimizer_and_scheduler
from dataset import create_dataloaders
import score
from score import MiaScore,MiaTrainerSettings


def main(args):

    # set loggers
    logger = logging.getLogger()

    # load model
    net = getattr(network, args.model.name)
    net = net(**args.model.model_args)
    if args.model.checkpoint is not None:
        net.load_state_dict(torch.load(
            args.model.checkpoint, map_location=args.DEVICE))
    net.name = args.model.name


    logger.info(f"Model has {count_parameters(net)} parameters")

    # load data
    dataloaders = create_dataloaders(config=args)

    # load trainer
    trainer_settings = TrainerSettings(name = net.name,optimizer=OptimizerConfig(**args.trainer.optimizer), scheduler=SchedulerConfig(**args.trainer.scheduler), log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.trainer.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    dataloaders = create_dataloaders(config=args)
    
    trainer = Trainer(model=net,dataloaders=dataloaders,trainer_args=trainer_settings)
    if args.trainer.train:
        trainer.train(epochs=args.trainer.epochs)

    # compute mia score
    for name, t, f, in [("retain_forget", dataloaders.retain, dataloaders.forget), ("forget_test", dataloaders.forget,  dataloaders.test), ("retain_test", dataloaders.retain, dataloaders.test)]:
        attack_model = getattr(score, args.attack_model.type)(**args.attack_model.model_args).to(args.device)
        mia_args = MiaTrainerSettings(optimizer=OptimizerConfig(**args.attack_trainer.optimizer), scheduler=SchedulerConfig(**args.attack_trainer.scheduler),**{k:v for k,v in args.attack_trainer.items() if k not in {"optimizer","scheduler","train"}})
        miaScore = MiaScore(parent_model=trainer.model, attack_model=attack_model,attack_train_settings=mia_args)
        mia_score = miaScore.compute_model_mia_score(t, f)
        logger.info(f"MIA score for {name} is {mia_score}")
    
    #insert unlearning algorithm here.

    return




def finetune_unleaerning(model, retain_loader, forget_loader):
    """
    #how forgetting happens in the model when finetuining.
    finetune_epochs = 100
    for lr in [1e-2,1e-3]:
        net = copy.deepcopy(trainer.model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4)# momentum=0.9, # the fine tuning loss should be very small.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
        finetune_trainer = Trainer(f"finetune_{lr}_"+net.name,net,retain_loader,val_loader,forget_loader,optimizer,scheduler,criterion,args.device)
        finetune_trainer.train(finetune_epochs,debug=True)
    """

def teacher_student_unlearning(model, retain_loader, forget_loader):

    full_model = copy.deepcopy(model)
    # train the model on the forget_loader in the
    return model


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
