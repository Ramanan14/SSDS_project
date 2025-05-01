import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = 5
num_increments = 5
classes_per_inc = 100 // num_increments
local_epochs = 5
batch_size = 64
lr = 0.01
dirichlet_alpha = 0.5

alpha_kd = 1.0
beta_ewc = 100.0
gamma_replay = 1.0
replay_size = 200

def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
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
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], 1)
        self.layer2 = self._make_layer(block, 32, layers[1], 2)
        self.layer3 = self._make_layer(block, 64, layers[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
])
full_train = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
full_test = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

def create_iid_splits():
    idxs = np.random.permutation(len(full_train))
    sz = len(idxs) // num_clients
    return [idxs[i*sz:(i+1)*sz].tolist() for i in range(num_clients)]

def create_noniid_splits(alpha):
    labels = np.array(full_train.targets)
    cls_idxs = {c: np.where(labels == c)[0] for c in range(100)}
    client_idxs = [[] for _ in range(num_clients)]
    for idxs in cls_idxs.values():
        props = np.random.dirichlet([alpha] * num_clients)
        cnt = (props * len(idxs)).astype(int)
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

def compute_fisher(model, loader):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        F.cross_entropy(model(x), y).backward()
        for n, p in model.named_parameters():
            fisher[n] += p.grad.data.pow(2)
    N = len(loader.dataset)
    return {n: fi / N for n, fi in fisher.items()}

def average_weights(ws):
    avg = copy.deepcopy(ws[0])
    n = len(ws)
    for k, v in avg.items():
        if v.dtype.is_floating_point:
            for w in ws[1:]:
                avg[k] += w[k]
            avg[k] /= n
    return avg

def evaluate(model, loader):
    model.eval()
    c = 0
    t = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1)
            c += (pred == y.to(device)).sum().item()
            t += y.size(0)
    return c / t

def run_glfc(split='iid'):
    client_idxs = create_iid_splits() if split == 'iid' else create_noniid_splits(dirichlet_alpha)
    states = [{'fisher': None, 'old_params': {}, 'old_global': None, 'exemplars': []}
              for _ in range(num_clients)]
    global_model = None
    results = []

    for inc in range(num_increments):
        seen = sum(tasks[:inc+1], [])
        new = tasks[inc]
        print(f"\n== Increment {inc+1}: new={new} ==")
        if global_model is None:
            global_model = ResNet32(len(seen)).to(device)
        else:
            global_model.fc = nn.Linear(global_model.fc.in_features, len(seen)).to(device)

        train_loader = get_loader(seen, True)
        fisher = compute_fisher(global_model, train_loader)
        old_params = {n: p.clone() for n, p in global_model.named_parameters()}

        for st in states:
            st['fisher'] = fisher
            st['old_params'] = old_params
            st['old_global'] = copy.deepcopy(global_model)

        test_loader = get_loader(seen, False)
        client_ws = []

        for cid, idxs in enumerate(client_idxs):
            my = [i for i in idxs if full_train.targets[i] in new]
            if not my:
                print(f"Client {cid}: no new data")
                continue
            loader = DataLoader(Subset(full_train, my), batch_size, shuffle=True)
            local_model = copy.deepcopy(global_model)
            opt = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
            st = states[cid]

            for _ in range(local_epochs):
                for x, y in loader:
                    opt.zero_grad()
                    out = local_model(x.to(device))
                    loss = F.cross_entropy(out, y.to(device))
                    for n, p in local_model.named_parameters():
                        if n in st['fisher'] and st['fisher'][n].shape == p.shape:
                            loss += beta_ewc * (st['fisher'][n] * (p - st['old_params'][n]).pow(2)).sum()
                    old_logits = st['old_global'](x.to(device))
                    k = old_logits.shape[1]
                    loss += alpha_kd * F.kl_div(
                        F.log_softmax(out[:, :k] / 2, dim=1),
                        F.softmax(old_logits / 2, dim=1),
                        reduction='batchmean'
                    )
                    if st['exemplars']:
                        xe, ye = zip(*st['exemplars'])
                        xe = torch.stack(xe).to(device)
                        ye = torch.tensor(ye).to(device)
                        loss += gamma_replay * F.cross_entropy(local_model(xe), ye)
                    loss.backward()
                    opt.step()
                    for xi, yi in zip(x.cpu(), y.cpu()):
                        st['exemplars'].append((xi, yi.item()))
                        if len(st['exemplars']) > replay_size:
                            st['exemplars'].pop(0)

            acc = evaluate(local_model, test_loader)
            print(f"Client {cid} acc: {acc:.4f}")
            client_ws.append(local_model.state_dict())

        if client_ws:
            global_model.load_state_dict(average_weights(client_ws))
        gacc = evaluate(global_model, test_loader)
        print(f"Global acc: {gacc:.4f}")
        results.append(gacc)

    return results

if __name__ == "__main__":
    print("IID GLFC:", run_glfc('iid'))
    print("Non-IID GLFC:", run_glfc('noniid'))
