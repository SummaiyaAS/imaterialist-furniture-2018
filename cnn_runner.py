import argparse
import shutil
import os

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import models
import utils
from utils import RunningMean
from misc import FurnitureDataset, preprocess, preprocess_with_augmentation, NB_CLASSES, preprocess_hflip

# For focal loss
from focal_loss import FocalLoss

BATCH_SIZE = 64
NUM_EPOCHS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    # Multi-GPU scaling
    if torch.cuda.device_count() > 1:
        print("Parallelizing model over %d GPUs!" % (torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    print('Model loaded successfully!')
    return model


def predict(model_name, outputDir):
    model = get_model(model_name)
    model_checkpoint = torch.load(os.path.join(outputDir, 'best_val_acc_weight_' + model_name + '.pth'))
    model.load_state_dict(model_checkpoint)
    model.eval()

    tta_preprocess = [preprocess, preprocess_hflip]

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('test', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders, device)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, os.path.join(outputDir, 'test_prediction_' + model_name + '.pth'))

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('val', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders, device)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, os.path.join(outputDir, 'val_prediction_' + model_name + '.pth'))


def train(model_name, outputDir):
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

    # Use model.fresh_params() to train only the newly initialized weights
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    if model_name.endswith("_focal"):
        print ("Using Focal loss instead of normal cross-entropy")
        criterion = FocalLoss(NB_CLASSES).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    min_loss = float("inf")
    max_acc = 0.0
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

        lx, px = utils.predict(model, validation_data_loader, device)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds == lx).float())
        error = torch.mean((preds != lx).float())
        print('Validation loss: %.5f | Accuracy: %.3f | Error: %.3f' % (log_loss, accuracy, error))
        scheduler.step(log_loss)

        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(outputDir, 'weight_' + model_name + '.pth'))

        betterModelFound = False
        if log_loss < min_loss:
            torch.save(model.state_dict(), os.path.join(outputDir, 'best_val_loss_weight_' + model_name + '.pth'))
            print('Validation score improved from %.5f to %.5f. Model snapshot saved!' % (min_loss, log_loss))
            min_loss = log_loss
            patience = 0
            betterModelFound = True

        if accuracy > max_acc:
            torch.save(model.state_dict(), os.path.join(outputDir, 'best_val_acc_weight_' + model_name + '.pth'))
            print('Validation accuracy improved from %.5f to %.5f. Model snapshot saved!' % (max_acc, accuracy))
            max_acc = accuracy
            patience = 0
            betterModelFound = True

        if not betterModelFound:
            patience += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('model', choices=['densenet', 'squeezenet', 'resnet', 'densenet_focal', 'squeezenet_focal'])
    parser.add_argument('--experiment', dest='experiment', action='store', type=str, default='01', help='Name of the experiment')
    args = parser.parse_args()
    outputDir = "Experiment_" + args.experiment + "_" + args.model
    print('Mode: %s | Model: %s | Output directory: %s' % (args.mode, args.model, outputDir))

    if args.mode == 'train':
        if os.path.exists(outputDir):
            print ("Removing previous experiment directory!")
            shutil.rmtree(outputDir)
        os.mkdir(outputDir)
        train(args.model, outputDir)
    elif args.mode == 'predict':
        predict(args.model, outputDir)
