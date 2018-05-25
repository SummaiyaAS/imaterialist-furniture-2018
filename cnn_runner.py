import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import models
import utils
from utils import RunningMean, use_gpu
from misc import FurnitureDataset, preprocess, preprocess_with_augmentation, NB_CLASSES, preprocess_hflip

# For focal loss
from focal_loss import FocalLoss

BATCH_SIZE = 32
NUM_EPOCHS = 20

def get_model(model_name):
    print('Loading model: %s' % (model_name))
    if model_name.startswith("densenet"):
        model = models.densenet201_finetune(NB_CLASSES)
    elif model_name.startswith("squeezenet"):
        model = models.squeezenet11_finetune(NB_CLASSES)
    elif model_name.startswith("resnet"):
        model = models.resnet152_finetune(NB_CLASSES)
    else:
        print("Error: Model not found!")
        exit(-1)
    print('done')
    return model


def predict(model_name):
    model = get_model(model_name)
    model.load_state_dict(torch.load('best_val_weight_' + model_name + '.pth'))
    model.eval()

    tta_preprocess = [preprocess, preprocess_hflip]

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('test', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, 'test_prediction_' + model_name + '.pth')

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('val', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, 'val_prediction_' + model_name + '.pth')


def train(model_name):
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation)
    val_dataset = FurnitureDataset('val', transform=preprocess)
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=12,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model(model_name)

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print('Number of learnable params: %s' % str(nb_learnable_params))

    # Multi-GPU scaling
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using %d GPUs!" % (torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    if model_name.endswith("_focal"):
        print ("Using Focal loss instead of normal cross-entropy")
        criterion = FocalLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    min_loss = float("inf")
    patience = 0
    for epoch in range(NUM_EPOCHS):
        print('Epoch: %d' % epoch)
        
        running_loss = RunningMean()
        running_error = RunningMean()
        running_accuracy = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.data[0], 1)
            running_error.update(torch.sum(preds != labels.data), batch_size)
            running_accuracy.update(torch.sum(preds == labels.data), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description('%.5f %.3f %.3f' % (running_loss.value, running_accuracy.value, running_error.value))
        print('Epoch: %d | Running loss: %.5f | Running accuracy: %.3f | Running error: %.3f' % (epoch, running_loss.value, running_accuracy.value, running_error.value))

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds == lx).float())
        error = torch.mean((preds != lx).float())
        print('Validation loss: %.5f | Accuracy: %.3f | Error: %.3f' % (log_loss, accuracy, error))
        scheduler.step(log_loss)

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'best_val_weight_' + model_name + '.pth')
            print('Validation score improved from %.5f to %.5f. Saved!' % (min_loss, log_loss))
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('model', choices=['densenet', 'squeezenet', 'resnet', 'squeezenet_focal'])
    args = parser.parse_args()
    print('Mode: %s | Model: %s' % (args.mode, args.model))
    if args.mode == 'train':
        train(args.model)
    elif args.mode == 'predict':
        predict(args.model)
