import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from Test import test

def train(args, train_data_loader, validate_data_loader, model, model_save_path):
    optimizer = SGD([
        {'params':model.get_network_params(), 'lr': args.lr_network},
        {'params':model.get_other_params(), 'lr':args.lr}
    ], weight_decay=args.wd, momentum=args.mo, nesterov=True)
    
    scheduler = MultiStepLR(optimizer, args.point, args.gamma)

    training_loss_list = []
    validating_accuracy_list = []
    best_validating_accuracy = 0

    training_loss = 0

    for task_index, task in enumerate(train_data_loader):
        model.train()

        images, labels = task
        images = images.float().cuda(args.devices[0])
        labels = labels.long().cuda(args.devices[0])

        loss = model.forward(images)
        training_loss += loss.cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (task_index + 1) % args.episode_gap == 0:
            training_loss /= args.episode_gap
            validating_accuracy = test(args, validate_data_loader, model)
            training_loss_list.append(training_loss)
            validating_accuracy_list.append(validating_accuracy)
            print('epoch %d finish: training loss = %f, validating acc = %f' % (
                (task_index + 1) / args.episode_gap, training_loss, validating_accuracy
            ))

            if not args.flag_debug:
                if validating_accuracy > best_validating_accuracy:
                    best_validating_accuracy = validating_accuracy
                    record = {
                        'state_dict': model.state_dict(),
                        'validating_accuracy': validating_accuracy,
                        'epoch': (task_index + 1) / args.episode_gap
                    }
                    torch.save(record, model_save_path)
            
            training_loss = 0
            scheduler.step()
    
    return training_loss_list, validating_accuracy_list



def train_st(args, train_data_loader, validate_data_loader, model, model_save_path, target_task):
    optimizer = SGD([
        {'params':model.get_network_params(), 'lr': args.lr_network},
        {'params':model.get_other_params(), 'lr':args.lr}
    ], weight_decay=args.wd, momentum=args.mo, nesterov=True)
    
    scheduler = MultiStepLR(optimizer, args.point, args.gamma)

    training_loss_list = []
    validating_accuracy_list = []
    best_validating_accuracy = 0

    training_loss = 0
    training_support_loss = 0
    training_distill_loss = 0
    n_taught_tasks = 0

    for task_index, task in enumerate(train_data_loader):
        model.train()

        images, labels = task
        images = images.float().cuda(args.devices[0])
        labels = labels.long().cuda(args.devices[0])
        
        label_set = labels[:args.N].cpu().numpy()
        flag_in_task = True
        for x in label_set:
            if not x in target_task:
                flag_in_task = False
                break
        
        if flag_in_task:
            n_taught_tasks += 1
            loss, support_loss, distill_loss = model.forward(images, output_type='loss', flag_in_task=True, labels=labels)
        else:
            loss = model.forward(images, output_type='loss')
        training_loss += loss.cpu().item()
        training_support_loss += support_loss.cpu().item()
        training_distill_loss += distill_loss.cpu().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (task_index + 1) % args.episode_gap == 0:
            training_loss /= args.episode_gap
            training_support_loss /= args.episode_gap
            training_distill_loss /= args.episode_gap
            validating_accuracy = test(args, validate_data_loader, model)
            training_loss_list.append(training_loss)
            validating_accuracy_list.append(validating_accuracy)
            print('epoch %d finish: training support loss = %f, training distill loss = %f, validating acc = %f, taught tasks = %d' % (
                    (task_index + 1) / args.episode_gap, training_support_loss, training_distill_loss, validating_accuracy, n_taught_tasks
                )
            )

            if not args.flag_debug:
                if validating_accuracy > best_validating_accuracy:
                    best_validating_accuracy = validating_accuracy
                    record = {
                        'state_dict': model.state_dict(),
                        'validating_accuracy': validating_accuracy,
                        'epoch': (task_index + 1) / args.episode_gap
                    }
                    torch.save(record, model_save_path)
            
            training_loss = 0
            n_taught_tasks = 0
            scheduler.step()
    
    return training_loss_list, validating_accuracy_list