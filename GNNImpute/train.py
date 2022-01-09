import os
import time
import glob
import torch


def train(gdata, model,
          no_cuda=False,
          epochs=3000,
          lr=0.001,
          weight_decay=0.0005,
          patience=200,
          fastmode=False,
          verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossFunc = torch.nn.MSELoss(reduction='mean')
    for key in gdata.keys():
        gdata[key] = gdata[key].to(device)

    def train_wrapper(epoch):
        model.train()
        optimizer.zero_grad()

        pred = model(gdata['x'], gdata['adj'], gdata['size_factors'])

        dropout_pred = pred[gdata['train_mask']]
        dropout_true = gdata['y'][gdata['train_mask']]

        loss_train = lossFunc(dropout_pred, dropout_true)

        loss_train.backward()
        optimizer.step()

        if not fastmode:
            model.eval()
            pred = model(gdata['x'], gdata['adj'], gdata['size_factors'])

        dropout_pred = pred[gdata['val_mask']]
        dropout_true = gdata['y'][gdata['val_mask']]

        loss_val = lossFunc(dropout_pred, dropout_true)

        if (epoch + 1) % 10 == 0 and verbose:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()))

        return loss_val.data.item()

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = float('inf')
    best_epoch = 0
    for epoch in range(epochs):
        loss_values.append(train_wrapper(epoch))

        if loss_values[-1] < best:
            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb != best_epoch:
            os.remove(file)

    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

    # Restore best model
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
