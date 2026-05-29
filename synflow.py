# import trainer

import train

from . import utils


def synflow(model, train_loader, test_loader, criterion, args):
    # optimizer, scheduler = trainer.get_optimizer_and_scheduler(model, args)


    # NOTE : commented above line and added below line 
    optimizer, scheduler = train.get_optimizer_and_scheduler(model, args)

    if args.rate != 0:
        utils.global_prune_model(model, args.rate, "synflow", train_loader)
        utils.check_sparsity(model)

    # trainer.train_with_rewind(

    # NOTE : commented above line added below line
    train.train_with_rewind(
        model, optimizer, scheduler, train_loader, criterion, args
    )
    utils.check_sparsity(model)
