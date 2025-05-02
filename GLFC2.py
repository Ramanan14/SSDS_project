import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

# ----------------------------
# 1) Setup & Hyperparams
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients     = 5
num_increments  = 5    # 100 classes → 5 increments of 20
classes_per_inc = 100 // num_increments
local_epochs    = 5
federated_rounds = 15  # ← 15 federated rounds per increment
batch_size      = 64
lr              = 0.01
dirichlet_alpha = 0.5

alpha_kd        = 1.0
beta_ewc        = 100.0
gamma_replay    = 1.0
replay_size     = 200

# ----------------------------
# 2) Model Definition (ResNet-32)
# ----------------------------
def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1    = conv3x3(in_c, out_c, stride)
        self.bn1      = nn.BatchNorm2d(out_c)
        self.conv2    = conv3x3(out_c, out_c)
        self.bn2      = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Identity()
        if stride!=1 or in_c!=out_c:
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
        self.conv1     = conv3x3(3,16)
        self.bn1       = nn.BatchNorm2d(16)
        self.layer1    = self._make_layer(block,16,layers[0],1)
        self.layer2    = self._make_layer(block,32,layers[1],2)
        self.layer3    = self._make_layer(block,64,layers[2],2)
        self.avgpool   = nn.AdaptiveAvgPool2d((1,1))
        self.fc        = nn.Linear(64*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride]+[1]*(num_blocks-1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); x = torch.flatten(x,1)
        return self.fc(x)

def ResNet32(num_classes):
    return ResNet(BasicBlock, [5,5,5], num_classes).to(device)

# ----------------------------
# 3) Data & Tasks
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4865,0.4409),(0.2009,0.1984,0.2023)),
])
full_train = datasets.CIFAR100("./data", train=True,  download=True, transform=transform)
full_test  = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

all_classes = list(range(100))
tasks = [ all_classes[i*classes_per_inc:(i+1)*classes_per_inc]
          for i in range(num_increments) ]

def create_iid_splits():
    idxs = np.random.permutation(len(full_train))
    sz   = len(idxs)//num_clients
    return [ idxs[i*sz:(i+1)*sz].tolist() for i in range(num_clients) ]

def create_noniid_splits(alpha):
    labels = np.array(full_train.targets)
    class_idxs = {c: np.where(labels==c)[0] for c in range(100)}
    client_idxs = [[] for _ in range(num_clients)]
    for idx_list in class_idxs.values():
        props    = np.random.dirichlet([alpha]*num_clients)
        counts   = (props * len(idx_list)).astype(int)
        counts[-1] = len(idx_list)-counts[:-1].sum()
        shuffled = np.random.permutation(idx_list)
        pos = 0
        for i,c in enumerate(counts):
            client_idxs[i].extend(shuffled[pos:pos+c])
            pos += c
    return client_idxs

def get_loader_for_classes(classes, train=True):
    ds   = full_train if train else full_test
    idxs = [i for i,l in enumerate(ds.targets) if l in classes]
    return DataLoader(Subset(ds,idxs), batch_size, shuffle=train)

# ----------------------------
# 4) CL Helper Losses
# ----------------------------
def compute_fisher(model, loader):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n,p in model.named_parameters()}
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        model.zero_grad()
        F.cross_entropy(model(x), y).backward()
        for n,p in model.named_parameters():
            fisher[n] += p.grad.data.pow(2)
    N = len(loader.dataset)
    return {n: fi/N for n,fi in fisher.items()}

def ewc_loss(model, old_params, fisher):
    loss = 0.0
    for n,p in model.named_parameters():
        loss += (fisher[n] * (p-old_params[n]).pow(2)).sum()
    return beta_ewc * loss

# ----------------------------
# 5) Centralized Upper‐Bounds
# ----------------------------
ub_accs = []
for inc in range(num_increments):
    seen = sum(tasks[:inc+1], [])
    central = ResNet32(len(seen))
    opt = optim.SGD(central.parameters(), lr=lr, momentum=0.9)
    loader_tr = get_loader_for_classes(seen, True)
    loader_te = get_loader_for_classes(seen, False)
    for _ in range(50):
        for x,y in loader_tr:
            opt.zero_grad()
            F.cross_entropy(central(x.to(device)), y.to(device)).backward()
            opt.step()
    ub = evaluate(central, loader_te)
    ub_accs.append(ub)
    print(f"[UB] Inc {inc+1}: Acc={ub:.4f}")

# ----------------------------
# 6) GLFC w/ 15 Rounds & Metrics
# ----------------------------
def run_glfc_with_metrics(split='iid'):
    client_idxs = create_iid_splits() if split=='iid' else create_noniid_splits(dirichlet_alpha)
    states = [{'fisher': None, 'old_params': {}, 'old_global': None, 'exemplars': []}
              for _ in range(num_clients)]
    global_model = None

    round_accs = []
    forgetting = []
    prev_final = []

    for inc in range(num_increments):
        seen = sum(tasks[:inc+1], [])
        new = tasks[inc]
        print(f"\n-- Inc {inc+1}/{num_increments}: new={new} --")

        if global_model is None:
            global_model = ResNet32(len(seen))
        else:
            global_model.fc = nn.Linear(global_model.fc.in_features, len(seen)).to(device)

        loader_cl = get_loader_for_classes(seen, True)
        Fm = compute_fisher(global_model, loader_cl)
        OP = {n: p.clone() for n, p in global_model.named_parameters()}
        for st in states:
            st['fisher'], st['old_params'] = Fm, OP
            st['old_global'] = copy.deepcopy(global_model)

        test_loader = get_loader_for_classes(seen, False)

        C = {n: torch.zeros_like(p, device=device) for n, p in global_model.named_parameters()}
        Ci = [{n: torch.zeros_like(p, device=device) for n, p in global_model.named_parameters()}
              for _ in range(num_clients)]

        this_rounds = []

        for rnd in range(1, federated_rounds+1):
            local_states = []
            for cid, idxs in enumerate(client_idxs):
                my = [i for i in idxs if full_train.targets[i] in new]
                if not my:
                    continue
                loader = DataLoader(Subset(full_train, my), batch_size, shuffle=True)
                local = copy.deepcopy(global_model)
                opt = optim.SGD(local.parameters(), lr=lr, momentum=0.9)
                st = states[cid]
                initW = {n: p.clone().to(device) for n, p in global_model.named_parameters()}

                for _ in range(local_epochs):
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        out = local(x)
                        loss = F.cross_entropy(out, y)
                        for n, p in local.named_parameters():
                            loss += beta_ewc * (st['fisher'][n] * (p - st['old_params'][n]).pow
