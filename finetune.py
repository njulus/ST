import os
import argparse
import random
import importlib

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from utils import global_variable as GV

def get_similarity_matrix(args, finetune_data_loader, baseline_model):
    n_classes = len(finetune_data_loader.dataset.label2name)
    if args.network_name == 'convnet':
        n_dimension = 64
    elif args.network_name == 'resnet':
        n_dimension = 640
    
    class_center = torch.zeros((n_classes, n_dimension))
    class_count = torch.zeros(n_classes)
    if not args.flag_not_use_gpu:
        class_center = class_center.cuda(args.devices[0])
        class_count = class_count.cuda(args.devices[0])
    
    for batch_index, batch in enumerate(finetune_data_loader):
        images, labels = batch
        images = images.float().cuda(args.devices[0]) if not args.flag_not_use_gpu else images.float()
        labels = labels.long().cuda(args.devices[0]) if not args.flag_not_use_gpu else labels.long()
        
        with torch.no_grad():
            embeddings = baseline_model.forward(images, flag_embedding=True)
            for i in range(0, n_classes):
                index_of_class_i = (labels == i)
                class_center[i] += torch.sum(embeddings[index_of_class_i], dim=0)
                class_count[i] += index_of_class_i.size()[0]
    class_count = class_count.unsqueeze(1)
    class_center = class_center / class_count
    class_center = F.normalize(class_center, p=2, dim=1)

    similarity_matrix = torch.mm(class_center, class_center.t())
    return class_center, similarity_matrix



def select_task(args, similarity_matrix):
    if args.data_name == 'mini_imagenet':
        n_classes = 64
    elif args.data_name == 'tiered_imagenet':
        n_classes = 351
    
    if args.policy == 'random':
        task = np.arange(n_classes)
        np.random.shuffle(task)
        task = task[:args.NN]
    elif args.policy == 'hardness':
        similarity_matrix = similarity_matrix.cpu().numpy()
        hardness_score = - np.sum(similarity_matrix, axis=1)
        task = np.argsort(hardness_score)[::-1][:args.NN].copy()
    return task



def get_task_indices(dataset, task):
    indices = []
    for label in task:
        indices += dataset.label2indices[label]
    return indices



def test_model(args, data_loader, model, task):
    label_map = torch.zeros(1 + max(task)) - 1
    label_map = label_map.long().cuda(args.devices[0])
    for i, x in enumerate(task):
        label_map[x] = i
    
    acc = 0
    model.eval()
    for batch_index, batch in enumerate(data_loader):
        images, labels = batch
        images = images.float().cuda(args.devices[0])
        labels = labels.long().cuda(args.devices[0])
        labels = label_map[labels]

        with torch.no_grad():
            logits = model(images)
        predicted_labels = torch.argmax(logits, dim=1)
        acc += torch.sum((predicted_labels == labels).float()).cpu().item()
    
    acc /= data_loader.dataset.__len__()
    return acc



def do_finetune(args, train_data_loader, validate_data_loader, finetune_model, task, finetune_model_save_path):
    optimizer = SGD([
        {'params':finetune_model.get_network_params(), 'lr': args.lr_network},
        {'params':finetune_model.get_other_params(), 'lr':args.lr}
    ], weight_decay=args.wd, momentum=args.mo, nesterov=True)
    
    scheduler = MultiStepLR(optimizer, args.point, args.gamma)

    label_map = torch.zeros(1 + max(task)) - 1
    label_map = label_map.long().cuda(args.devices[0])
    for i, x in enumerate(task):
        label_map[x] = i

    best_validating_accuracy = 0

    for epoch in range(0, args.n_training_epochs):
        finetune_model.train()
        training_loss = 0
        training_acc = 0
        for batch_index, batch in enumerate(train_data_loader):
            images, labels = batch
            images = images.float().cuda(args.devices[0])
            labels = labels.long().cuda(args.devices[0])
            labels = label_map[labels]

            logits = finetune_model(images)
            loss_value = nn.CrossEntropyLoss()(logits, labels)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            predicted_labels = torch.argmax(logits, dim=1)
            training_loss += loss_value.cpu().item()
            training_acc += torch.sum((predicted_labels == labels).float()).cpu().item()

        training_loss /= train_data_loader.dataset.__len__()
        training_acc /= train_data_loader.dataset.__len__()
        finetune_model.eval()
        validation_acc = test_model(args, validate_data_loader, finetune_model, task)
        print('finetune epoch %d finish: loss = %f, tr-acc = %f, va-acc = %f' % 
            (epoch + 1, training_loss, training_acc, validation_acc))

        if not args.flag_debug:
            if validation_acc > best_validating_accuracy:
                best_validating_accuracy = validation_acc
                record = {
                    'state_dict': finetune_model.state_dict(),
                    'validating_accuracy': validation_acc,
                    'epoch': epoch + 1,
                    'task': task,
                    'args': args
                }
                torch.save(record, finetune_model_save_path)

        scheduler.step()

    return best_validating_accuracy



def display_args(args):
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('network_name = %s' % (args.network_name))
    print('model_name = %s' % (args.model_name))
    print('NN = %d' % (args.NN))
    print('===== experiment environment arguments =====')
    print('devices = %s' % str(args.devices))
    print('flag_debug = %r' % (args.flag_debug))
    print('n_workers = %d' % (args.n_workers))
    print('===== optimizer arguments =====')
    print('lr_network = %f' % (args.lr_network))
    print('lr = %f' % (args.lr))
    print('point = %s' % str(args.point))
    print('gamma = %f' % (args.gamma))
    print('wd = %f' % (args.wd))
    print('mo = %f' % (args.mo))
    print('===== training procedure arguments =====')
    print('policy = %s' % (args.policy))
    print('n_training_epochs = %d' % (args.n_training_epochs))
    print('batch_size = %d' % (args.batch_size))
    


if __name__ == '__main__':
    # set random seed
    random.seed(960402)
    np.random.seed(960402)
    torch.manual_seed(960402)
    torch.cuda.manual_seed(960402)
    torch.backends.cudnn.deterministic = True

    # create a parser
    parser = argparse.ArgumentParser()
    # task arguments
    parser.add_argument('--data_name', type=str, default='mini_imagenet', choices=['mini_imagenet', 'tiered_imagenet'])
    parser.add_argument('--network_name', type=str, default='resnet', choices=['resnet'])
    parser.add_argument('--model_name', type=str, default='linear_classifier')
    parser.add_argument('--NN', type=int, default=64) # number of selected classes
    # experiment environment arguments
    parser.add_argument('--devices', type=int, nargs='+', default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
    # optimizer arguments
    parser.add_argument('--lr_network', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--point', type=int, nargs='+', default=(20,30,40))
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=0.0005)  # weight decay
    parser.add_argument('--mo', type=float, default=0.9)  # momentum
    # training procedure arguments
    parser.add_argument('--policy', type=str, default='hardness', choices=['random', 'hardness'])
    parser.add_argument('--n_training_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()

    display_args(args)

    data_path = 'datasets/' + args.data_name + '/'

    # import modules
    Data = importlib.import_module('dataloaders.' + args.data_name)
    Network = importlib.import_module('networks.' + args.network_name)
    Model = importlib.import_module('models.' + args.model_name)

    # generate datasets and data loaders
    finetune_dataset = Data.MyDataset(data_path, 'finetune')
    finetune_data_loader = DataLoader(finetune_dataset, args.batch_size, shuffle=True, drop_last=False)
    auxiliary_dataset = Data.MyDataset(data_path, 'auxiliary')
    auxiliary_data_loader = DataLoader(auxiliary_dataset, args.batch_size, shuffle=True, drop_last=False)
    print('===== dataset and data loader ready. =====')

    # compute similarity matrix with baseline model
    similarity_matrix_path = 'saves/similarity_matrices/' + \
        args.data_name + '.sim'
    if os.path.exists(similarity_matrix_path):
        similarity_matrix = torch.load(similarity_matrix_path)['similarity']
    else:
        # generate baseline network
        baseline_network = Network.MyNetwork(args)
        if len(args.devices) > 1:
            baseline_network = torch.nn.DataParallel(baseline_network, device_ids=args.devices)
        print('===== baseline network ready. =====')

        # generate baseline model
        if args.data_name == 'mini_imagenet':
            out_dimension = 64
        elif args.data_name == 'tiered_imagenet':
            out_dimension = 351
        baseline_model = Model.MyModel(args, baseline_network, out_dimension)
        pretrained_file_path = 'saves/pretrained_weights/' + args.data_name + '/' + args.network_name + '.pth'
        pretrained_state_dict = torch.load(pretrained_file_path)['params']
        pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if k.startswith('encoder')}
        baseline_model_state_dict = baseline_model.state_dict()
        baseline_model_state_dict.update(pretrained_state_dict)
        baseline_model.load_state_dict(baseline_model_state_dict)
        baseline_model = baseline_model.cuda(args.devices[0])
        print('===== baseline model ready. =====')

        class_center, similarity_matrix = get_similarity_matrix(args, finetune_data_loader, baseline_model)
        sim_dict = {
            'center': class_center,
            'similarity': similarity_matrix
        }
        torch.save(sim_dict, similarity_matrix_path)
    print('===== similarity matrix ready. =====')

    # perform task selection
    task = select_task(args, similarity_matrix)
    train_dataset = Subset(finetune_dataset, indices=get_task_indices(finetune_dataset, task))
    train_data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=False)
    validate_dataset = Subset(auxiliary_dataset, indices=get_task_indices(auxiliary_dataset, task))
    validate_data_loader = DataLoader(validate_dataset, args.batch_size, shuffle=True, drop_last=False)

    # generate finetune network
    finetune_network = Network.MyNetwork(args)
    if len(args.devices) > 1:
        finetune_network = torch.nn.DataParallel(finetune_network, device_ids=args.devices)
    print('===== finetune network ready. =====')

    # generate finetune model
    out_dimension = args.NN
    finetune_model = Model.MyModel(args, finetune_network, out_dimension)
    pretrained_file_path = 'saves/pretrained_weights/' + args.data_name + '/' + args.network_name + '.pth'
    pretrained_state_dict = torch.load(pretrained_file_path)['params']
    finetune_model_state_dict = finetune_model.state_dict()
    fc_weight = pretrained_state_dict['fc.weight'][task, :]
    fc_bias = pretrained_state_dict['fc.bias'][task]
    pretrained_state_dict['fc.weight'] = fc_weight
    pretrained_state_dict['fc.bias'] = fc_bias
    finetune_model.load_state_dict(pretrained_state_dict)
    finetune_model = finetune_model.cuda(args.devices[0])
    print('===== finetune model ready. =====')

    finetune_model_save_path = 'saves/finetuned_models/' + args.data_name + '/' + \
        args.network_name + \
        '_NN=' + str(args.NN) + \
        '_lr-net=' + str(args.lr_network) + \
        '_lr=' + str(args.lr) + \
        '_point=' + str(args.point) + \
        '_gamma=' + str(args.gamma) + \
        '_wd=' + str(args.wd) + \
        '_mo=' + str(args.mo) + \
        '_policy=' + str(args.policy) + \
        '.model'
    
    init_train_acc = test_model(args, train_data_loader, finetune_model, task)
    init_val_acc = test_model(args, validate_data_loader, finetune_model, task)
    print('baseline train acc = %f' % (init_train_acc))
    print('baseline val acc = %f' % (init_val_acc))

    # perform finetuning
    finetune_val_acc = do_finetune(args, train_data_loader, validate_data_loader,
        finetune_model, task, finetune_model_save_path)
    print('finetune val acc = %f' % (finetune_val_acc))
