import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

# ----------------------------
# Hyperparameters & Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_clients     = 5
num_increments  = 5
classes_per_inc = 100 // num_increments
local_epochs    = 5
batch_size      = 64
lr              = 0.01
dirichlet_alpha = 0.5
replay_size     = 200
num_rounds      = 50

gamma_replay = 1.0  # weight for exemplar replay
alpha_gc     = 1.0  # weight for gradient compensation loss
alpha_rd     = 1.0  # weight for relation distillation loss
T            = 2.0  # temperature for distillation
epsilon      = 1e-8 # numerical stability for weights

# ----------------------------
# Helper functions
# ----------------------------
def average_weights(ws):
    avg = copy.deepcopy(ws[0])
    n   = len(ws)
    for k, v in avg.items():
        if v.dtype.is_floating_point:
            for w in ws[1:]: avg[k] += w[k]
            avg[k] /= n
    return avg


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total if total > 0 else 0

# ----------------------------
# Model Definitions
# ----------------------------
def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2   = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], 1)
        self.layer2 = self._make_layer(block, 32, layers[1], 2)
        self.layer3 = self._make_layer(block, 64, layers[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(64 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def ResNet32(num_classes):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)

# ----------------------------
# Data & Splits
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
])
full_train = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
full_test  = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

def create_iid_splits():
    idxs = np.random.permutation(len(full_train))
    sz   = len(idxs) // num_clients
    return [idxs[i*sz:(i+1)*sz].tolist() for i in range(num_clients)]

def create_noniid_splits(alpha):
    labels = np.array(full_train.targets)
    cls_idxs = {c: np.where(labels == c)[0] for c in range(100)}
    client_idxs = [[] for _ in range(num_clients)]
    for idxs in cls_idxs.values():
        props = np.random.dirichlet([alpha] * num_clients)
        cnt   = (props * len(idxs)).astype(int)
        cnt[-1] = len(idxs) - cnt[:-1].sum()
        sh = np.random.permutation(idxs)
        pos = 0
        for i, c in enumerate(cnt):
            client_idxs[i].extend(sh[pos:pos+c])
            pos += c
    return client_idxs

all_classes = list(range(100))
tasks = [all_classes[i*classes_per_inc:(i+1)*classes_per_inc] for i in range(num_increments)]

def get_loader(classes, train=True):
    ds = full_train if train else full_test
    idxs = [i for i, l in enumerate(ds.targets) if l in classes]
    return DataLoader(Subset(ds, idxs), batch_size, shuffle=train)

# ----------------------------
# Federated Training Loop
# ----------------------------
def run_glfc(split='iid'):
    client_idxs = create_iid_splits() if split == 'iid' else create_noniid_splits(dirichlet_alpha)
    states = [{'old_global': None, 'exemplars': []} for _ in range(num_clients)]
    global_model = None
    global_results = []
    forgetting_rates = []
    last_gacc = None

    for inc in range(num_increments):
        seen = sum(tasks[:inc+1], [])
        new  = tasks[inc]
        # init or expand model
        if global_model is None:
            global_model = ResNet32(len(seen)).to(device)
        else:
            global_model.fc = nn.Linear(global_model.fc.in_features, len(seen)).to(device)

        # snapshot old global for distillation
        for st in states:
            st['old_global'] = copy.deepcopy(global_model)

        # mapping class labels to output indices
        mapping = torch.full((100,), -1, dtype=torch.long, device=device)
        mapping[seen] = torch.arange(len(seen), device=device)
        old_ids = sum(tasks[:inc], [])
        old_ids_tensor = torch.tensor(old_ids, device=device) if old_ids else None

        # local updates
        for _ in range(num_rounds):
            client_ws = []
            for cid, idxs in enumerate(client_idxs):
                my_idxs = [i for i in idxs if full_train.targets[i] in new]
                if not my_idxs: continue
                loader = DataLoader(Subset(full_train, my_idxs), batch_size, shuffle=True)
                local_model = copy.deepcopy(global_model)
                opt = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
                st  = states[cid]

                for _ in range(local_epochs):
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        out = local_model(x)

                        # -- Gradient Compensation --
                        probs    = F.softmax(out, dim=1)
                        grad_mag = torch.abs(probs[torch.arange(y.size(0)), y] - 1.0)
                        if old_ids:
                            mask_old = torch.isin(y, old_ids_tensor)
                        else:
                            mask_old = torch.zeros_like(y, dtype=torch.bool)
                        mask_new = ~mask_old
                        mean_old = grad_mag[mask_old].mean() if mask_old.any() else grad_mag.mean()
                        mean_new = grad_mag[mask_new].mean() if mask_new.any() else grad_mag.mean()
                        weights  = torch.zeros_like(grad_mag)
                        weights[mask_old] = grad_mag[mask_old] / (mean_old + epsilon)
                        weights[mask_new] = grad_mag[mask_new] / (mean_new + epsilon)
                        ce_batch = F.cross_entropy(out, y, reduction='none')
                        loss_gc  = (weights * ce_batch).mean()

                        # -- Relation Distillation --
                        with torch.no_grad():
                            old_logits = st['old_global'](x)
                            p_old      = F.softmax(old_logits / T, dim=1)
                        C_old = old_logits.size(1)
                        target = torch.zeros_like(out)
                        target[:, :C_old] = p_old
                        y_idx = mapping[y]
                        target[torch.arange(y.size(0)), y_idx] = 1.0
                        target = target / target.sum(dim=1, keepdim=True)
                        log_p_new = F.log_softmax(out / T, dim=1)
                        loss_rd   = F.kl_div(log_p_new, target, reduction='batchmean') * (T * T)

                        # combine main losses
                        loss = alpha_gc * loss_gc + alpha_rd * loss_rd

                        # -- Exemplar Replay --
                        if st['exemplars']:
                            xe, ye = zip(*st['exemplars'])
                            xe = torch.stack(xe).to(device)
                            ye = torch.tensor(ye, device=device)
                            loss += gamma_replay * F.cross_entropy(local_model(xe), ye)

                        loss.backward()
                        opt.step()

                        # update exemplars
                        for xi, yi in zip(x.cpu(), y.cpu()):
                            st['exemplars'].append((xi, yi.item()))
                            if len(st['exemplars']) > replay_size:
                                st['exemplars'].pop(0)

                client_ws.append(local_model.state_dict())
            if client_ws:
                global_model.load_state_dict(average_weights(client_ws))

        # -- Evaluation --
        seen_loader = get_loader(seen, train=False)
        gacc = evaluate(global_model, seen_loader)
        global_results.append(gacc)
        if last_gacc is not None and old_ids:
            old_loader = get_loader(old_ids, train=False)
            acc_old    = evaluate(global_model, old_loader)
            forgetting_rates.append((last_gacc - acc_old) / last_gacc)
        last_gacc = gacc

    avg_forgetting = sum(forgetting_rates) / len(forgetting_rates) if forgetting_rates else 0
    print("Global acc per inc:", global_results)
    print("Avg forgetting rate:", avg_forgetting)


if __name__ == "__main__":
    run_glfc('iid')
    run_glfc('noniid')

