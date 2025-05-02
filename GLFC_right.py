import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

# Setup device and parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = 5
num_tasks = 5
classes_per_task = 100 // num_tasks
local_epochs = 5
batch_size = 64
learning_rate = 0.01
dir_alpha = 0.5
replay_capacity = 200
rounds = 50

# Loss weights and temperature
gamma_replay = 1.0
alpha_gc = 1.0
alpha_rd = 1.0
T = 2.0
epsilon = 1e-8

# Helper functions
def average_weights(weights_list):
    avg = copy.deepcopy(weights_list[0])
    for key in avg:
        if avg[key].dtype.is_floating_point:
            for w in weights_list[1:]:
                avg[key] += w[key]
            avg[key] /= len(weights_list)
    return avg


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0

# Model definition
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, layers, num_classes):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, layers[0], stride=1)
        self.layer2 = self._make_layer(32, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(BasicBlock(self.in_planes, out_channels, s))
            self.in_planes = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Data loaders and splits
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023))
])
train_set = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

def iid_splits():
    idxs = np.random.permutation(len(train_set))
    size = len(idxs) // num_clients
    return [idxs[i*size:(i+1)*size].tolist() for i in range(num_clients)]

def noniid_splits(alpha):
    labels = np.array(train_set.targets)
    idx_per_class = {c: np.where(labels == c)[0] for c in range(100)}
    client_idxs = [[] for _ in range(num_clients)]
    for idxs in idx_per_class.values():
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(idxs)).astype(int)
        counts[-1] = len(idxs) - counts[:-1].sum()
        shuffled = np.random.permutation(idxs)
        pointer = 0
        for i, cnt in enumerate(counts):
            client_idxs[i].extend(shuffled[pointer:pointer+cnt])
            pointer += cnt
    return client_idxs

all_classes = list(range(100))
tasks = [all_classes[i*classes_per_task:(i+1)*classes_per_task] for i in range(num_tasks)]

def get_loader(classes, train=True):
    dataset = train_set if train else test_set
    idxs = [i for i, label in enumerate(dataset.targets) if label in classes]
    return DataLoader(Subset(dataset, idxs), batch_size, shuffle=train)

# Federated GLFC

def run_glfc(split='iid'):
    client_idxs = iid_splits() if split == 'iid' else noniid_splits(dir_alpha)
    states = [{'old_global': None, 'exemplars': []} for _ in range(num_clients)]
    global_model = None
    global_accs, forget_rates = [], []

    for t in range(num_tasks):
        seen = sum(tasks[:t+1], [])
        new = tasks[t]
        if global_model is None:
            global_model = ResNet([5,5,5], len(seen)).to(device)
        else:
            global_model.fc = nn.Linear(global_model.fc.in_features, len(seen)).to(device)

        for state in states:
            state['old_global'] = copy.deepcopy(global_model)

        mapping = torch.full((100,), -1, device=device, dtype=torch.long)
        mapping[seen] = torch.arange(len(seen), device=device)
        old_ids = sum(tasks[:t], [])
        old_ids_tensor = torch.tensor(old_ids, device=device) if old_ids else None

        for _ in range(rounds):
            client_weights = []
            for cid, idxs in enumerate(client_idxs):
                sample_idxs = [i for i in idxs if train_set.targets[i] in new]
                if not sample_idxs:
                    continue
                loader = DataLoader(Subset(train_set, sample_idxs), batch_size, shuffle=True)
                local_model = copy.deepcopy(global_model)
                optimizer = optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
                state = states[cid]

                for _ in range(local_epochs):
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        out = local_model(x)

                        # gradient compensation
                        probs = F.softmax(out, dim=1)
                        grad_mag = (probs[torch.arange(len(y)), y] - 1).abs()
                        mask_old = torch.isin(y, old_ids_tensor) if old_ids else torch.zeros_like(y, dtype=torch.bool)
                        mask_new = ~mask_old
                        mean_old = grad_mag[mask_old].mean() if mask_old.any() else grad_mag.mean()
                        mean_new = grad_mag[mask_new].mean() if mask_new.any() else grad_mag.mean()
                        weights = torch.zeros_like(grad_mag)
                        weights[mask_old] = grad_mag[mask_old] / (mean_old + epsilon)
                        weights[mask_new] = grad_mag[mask_new] / (mean_new + epsilon)
                        ce_loss = (weights * F.cross_entropy(out, y, reduction='none')).mean()

                        # relation distillation
                        with torch.no_grad():
                            old_logits = state['old_global'](x)
                            p_old = F.softmax(old_logits / T, dim=1)
                        C_old = old_logits.size(1)
                        target = torch.zeros_like(out)
                        target[:, :C_old] = p_old
                        target[torch.arange(len(y)), mapping[y]] = 1
                        target /= target.sum(dim=1, keepdim=True)
                        loss_rd = F.kl_div(F.log_softmax(out / T, dim=1), target, reduction='batchmean') * T * T

                        loss = alpha_gc * ce_loss + alpha_rd * loss_rd
                        if state['exemplars']:
                            xe, ye = zip(*state['exemplars'])
                            xe = torch.stack(xe).to(device)
                            ye = torch.tensor(ye, device=device)
                            loss += gamma_replay * F.cross_entropy(local_model(xe), ye)

                        loss.backward()
                        optimizer.step()

                        for xi, yi in zip(x.cpu(), y.cpu()):
                            state['exemplars'].append((xi, yi.item()))
                            if len(state['exemplars']) > replay_capacity:
                                state['exemplars'].pop(0)

                client_weights.append(local_model.state_dict())
            if client_weights:
                global_model.load_state_dict(average_weights(client_weights))

        acc_seen = evaluate(global_model, get_loader(seen, train=False))
        global_accs.append(acc_seen)
        if old_ids:
            acc_old = evaluate(global_model, get_loader(old_ids, train=False))
            forget_rates.append((global_accs[-2] - acc_old) / global_accs[-2])

    print("Accuracies:", global_accs)
    print("Forgetting:", sum(forget_rates)/len(forget_rates) if forget_rates else 0)

if __name__ == "__main__":
    run_glfc('iid')
    run_glfc('noniid')


