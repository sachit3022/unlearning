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
import trainer as tr
from trainer import Trainer, TrainerSettings, count_parameters, get_optimizer_and_scheduler
from dataset import create_dataloaders,get_finetune_dataloaders,create_dataloaders_missing_class
import score
from score import MiaScore,MiaTrainerSettings

def compute_unlearning_metrics(args, model, dataloaders):
    mia_scores = {"retain_forget":0,"forget_test":0,"retain_test":0}
    for name, t, f, in [("retain_forget", dataloaders.retain, dataloaders.forget), ("forget_test", dataloaders.forget,  dataloaders.test), ("retain_test", dataloaders.retain, dataloaders.test)]:
        attack_model = getattr(score, args.attack_model.type)(**args.attack_model.model_args).to(args.device)
        attack_optimizer_config = getattr(tr, args.attack_trainer.optimizer.type + "OptimConfig" )(**args.attack_trainer.optimizer)
        attack_scheduler_config = getattr(tr, args.attack_trainer.scheduler.type + "SchedulerConfig" )(**args.attack_trainer.scheduler)
        mia_args = MiaTrainerSettings(optimizer=attack_optimizer_config,log_path=args.directory.LOG_PATH,model_dir=args.MODEL_DIR, scheduler=attack_scheduler_config,**{k:v for k,v in args.attack_trainer.items() if k not in {"optimizer","scheduler","train"}})
        miaScore = MiaScore(parent_model=model, attack_model=attack_model,charecterstic = args.attack_model.charecterstic,folds=args.attack_model.folds, attack_train_settings=mia_args)
        mia_scores[name] = miaScore.compute_model_mia_score(t, f)
    return mia_scores


def finetune_unlearning(args, model, dataloaders):
    #how forgetting happens in the model when finetuining.
    optimizer_config = getattr(tr, args.finetune.optimizer.type + "OptimConfig" )(**args.finetune.optimizer)
    scheduler_config = getattr(tr, args.finetune.scheduler.type + "SchedulerConfig" )(**args.finetune.scheduler)
    trainer_settings = TrainerSettings(name = f"finetune_{model.name}",optimizer=optimizer_config, scheduler=scheduler_config, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.trainer.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    finetune_dataloaders = get_finetune_dataloaders(dataloaders)
    finetune_trainer = Trainer(model=model,dataloaders=finetune_dataloaders,trainer_args=trainer_settings)
    finetune_trainer.train(args.finetune.epochs)
    return finetune_trainer.model

    

def teacher_student_unlearning(model, retain_loader, forget_loader):

    full_model = copy.deepcopy(model)
    # train the model on the forget_loader in the
    return model

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

    # load data
    dataloaders = create_dataloaders_missing_class(config=args)

    # load trainer
    optimizer_config = getattr(tr, args.trainer.optimizer.type + "OptimConfig" )(** args.trainer.optimizer)
    scheduler_config = getattr(tr, args.trainer.scheduler.type + "SchedulerConfig" )(**args.trainer.scheduler)

    trainer_settings = TrainerSettings(name = net.name,optimizer=optimizer_config, scheduler=scheduler_config, log_path= args.directory.LOG_PATH,device=args.device, **{k:v for k,v in args.trainer.items() if k not in {"optimizer","scheduler","train","epochs"}} )
    
    trainer = Trainer(model=net,dataloaders=dataloaders,trainer_args=trainer_settings)
    if args.trainer.train:
        trainer.train(epochs=args.trainer.epochs)

    # compute mia score
    mia_scores = compute_unlearning_metrics(args, trainer.model, dataloaders)
    logger.info(f"MIA score before unlearning is {mia_scores}")
    
    #insert unlearning algorithm here.
    unlearned_model = finetune_unlearning(args, trainer.model, dataloaders)

    # compute after unlearning mia score
    mia_scores = compute_unlearning_metrics(args, unlearned_model, dataloaders)
    logger.info(f"MIA score after unlearning for is {mia_scores}")

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
