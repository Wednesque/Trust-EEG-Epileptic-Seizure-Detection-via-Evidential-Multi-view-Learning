import os
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import MultiViewDataset
from models import EML


def train(device='cuda' if torch.cuda.is_available() else 'cpu',
          train_path='dataset/eeg/data/domain_feature/data2_fold0_train.pkl',
          valid_path='dataset/eeg/data/domain_feature/data1_fold0_valid.pkl',
          #   train_path = 'dataset/handwritten_6views_train.pkl',
          #   valid_path = 'dataset/handwritten_6views_test.pkl',
          epochs=20,
          saving_path=None):
    # Load dataset
    data_train = MultiViewDataset(data_path=train_path)
    data_valid = MultiViewDataset(data_path=valid_path)
    num_classes = len(set(data_train.y))
    train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=1024, shuffle=False)

    # Define model
    model = EML(sample_shapes=[s.shape for s in data_train[0]['x'].values()], num_classes=num_classes).to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.1)

    best_valid_acc = 0.
    best_model_wts = model.state_dict()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, num_samples = 0, 0, 0
        for batch in train_loader:
            x, target = batch['x'], batch['y']
            for v in x.keys():
                x[v] = x[v].to(device)
            target = target.to(device)
            view_e, fusion_e, loss = model(x, target, kl_penalty=min(1., epoch / 20))
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            train_loss += loss.mean().item() * len(target)
            correct += torch.sum(fusion_e.argmax(dim=-1).eq(target)).item()
            num_samples += len(target)
        scheduler.step()
        train_loss = train_loss / num_samples
        train_acc = correct / num_samples
        val = validate(device, model, valid_loader)
        if best_valid_acc < val['accuracy']:
            best_valid_acc = val['accuracy']
            best_model_wts = copy.deepcopy(model.state_dict())  # save the best model
        print(f'Epoch {epoch:2d}; train loss {train_loss:.4f}, train acc {train_acc:.4f};', end=' ')
        if num_classes == 2:
            print('validation:', *[f'{v:.4f}' for k, v in val.items() if k.startswith('b_')])
        else:
            print('validation:', val['accuracy'])

    model.load_state_dict(best_model_wts)
    val = validate(device, model, valid_loader)
    print('Validation for best model:', *[f'{k}:{v:.6f}' for k, v in val.items()])
    if saving_path is not None:
        os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        torch.save(model, saving_path)
    return model


def validate(device, model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = num_samples = 0  # For multi classification
        TP = TN = FP = FN = 0  # For binary classification
        for batch in dataloader:
            x, y = batch['x'], batch['y']
            for v in x.keys():
                x[v] = x[v].to(device)
            view_e, fusion_e, loss = model(x)
            pred = fusion_e.cpu().argmax(dim=-1)
            correct += torch.sum(pred == y).item()
            num_samples += len(y)
            TP += torch.sum((pred == 1) & (pred == y)).item()
            TN += torch.sum((pred == 0) & (pred == y)).item()
            FP += torch.sum((pred == 1) & (pred != y)).item()
            FN += torch.sum((pred == 0) & (pred != y)).item()
    accuracy = correct / num_samples
    b_accuracy = (TN + TP) / (TP + TN + FP + FN)
    b_sensitivity = TP / (TP + FN)
    b_specificity = TN / (TN + FP)
    return {
        'accuracy': accuracy,
        'b_accuracy': b_accuracy, 'b_sensitivity': b_sensitivity, 'b_specificity': b_specificity
    }


if __name__ == '__main__':
    train(saving_path='model/model.pt')
