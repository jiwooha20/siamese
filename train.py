import argparse
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import pickle
import torch
from torch.utils.data import DataLoader
from utils import musicDatasetTest, musicDatasetTrain
from model import Siamese_mfcc
from torch.autograd import Variable
warnings.filterwarnings('ignore')


def main(args):
    path = "./save_dir_"
    
    # hyperparameter
    exp_name = args.exp_name
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    lr_scheduler = args.lr_scheduler
    data_type = args.data_type
    max_iter = args.max_iter
    show_every = args.show_every
    save_every = args.save_every
    test_every = args.test_every

    os.makedirs(f"./siamase/{exp_name}", exist_ok=True)
    model_path = f"./siamase/{exp_name}"

    
    path+data_type+"/"
    train_dataset = musicDatasetTrain(path+data_type+"/", 10000)
    test_dataset = musicDatasetTrain(path+data_type+"/", 10000)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Test Data Size : {len(test_dataset)}")

    # 학습 장치
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)

    num_classes = 3  # 2 class (normal, TIL) + background
    model = Siamese_mfcc()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0001)
    optimizer.zero_grad()

    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        scheduler = None

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    for batch_id, (img1, img2, label) in enumerate(train_dataloader, 1):
        if batch_id > max_iter:
            break
        if torch.cuda.is_available():
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        optimizer.zero_grad()
        #print(img1.shape)
        #print(img2.shape)
        output = model.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % save_every == 0:
            torch.save(model.state_dict(), model_path + '/model-inter-' + str(batch_id+1) + ".pt")
        if batch_id % test_every == 0:
            right, error = 0, 0
            for _, (test1, test2) in enumerate(test_dataloader, 1):
                if torch.cuda.is_available():
                    test1, test2 = test1.cuda(), test2.cuda()
                test1, test2 = Variable(test1), Variable(test2)
                output = model.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else: error += 1
            print('*'*70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*70)
            queue.append(right*1.0/(right+error))
        train_loss.append(loss_val)
    #  learning_rate = learning_rate * 0.95

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#"*70)
    print("final accuracy: ", acc/20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_scheduler", action="store_false")
    parser.add_argument("--data_type",type=str,default="mfcc")
    parser.add_argument("--show_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--test_every", type=int, default=100)
    args = parser.parse_args()
    print(args)
    main(args)