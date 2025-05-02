import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

# ----------------------------
# 1) Hyperparameters & Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = 5
num_increments = 5
classes_per_inc = 100 // num_increments
local_epochs = 5
batch_size = 64
lr = 0.01
dirichlet_alpha = 0.5
federated_rounds = 15

# Centralized upper-bound accuracies (precomputed) for increments 1..5
centralized_ub = [0.4955, 0.4745, 0.4758, 0.4445, 0.4581]

alpha_kd = 1.0
beta_ewc = 100.0

# ----------------------------
# 2) Model Definition
# ----------------------------

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
        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block,16,layers[0],1)
        self.layer2 = self._make_layer(block,32,layers[1],2)
        self.layer3 = self._make_layer(block,64,layers[2],2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1]*(num_blocks-1):
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
# 3) Data & Splits
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4865,0.4409),(0.2009,0.1984,0.2023)),
])
full_train = datasets.CIFAR100("./data",True,transform=transform,download=True)
full_test  = datasets.CIFAR100("./data",False,transform=transform,download=True)

def create_iid_splits():
    idxs = np.random.permutation(len(full_train))
    size = len(idxs)//num_clients
    return [idxs[i*size:(i+1)*size].tolist() for i in range(num_clients)]

def create_noniid_splits(alpha):
    labels = np.array(full_train.targets)
    cls_idxs = {c: np.where(labels==c)[0] for c in range(100)}
    client_idxs = [[] for _ in range(num_clients)]
    for idx_list in cls_idxs.values():
        props = np.random.dirichlet([alpha]*num_clients)
        counts = (props * len(idx_list)).astype(int)
        counts[-1] = len(idx_list)-counts[:-1].sum()
        sh = np.random.permutation(idx_list)
        pos=0
        for i,cnt in enumerate(counts):
            client_idxs[i].extend(sh[pos:pos+cnt])
            pos+=cnt
    return client_idxs

all_classes = list(range(100))
tasks = [all_classes[i*classes_per_inc:(i+1)*classes_per_inc] for i in range(num_increments)]

def get_loader(idxs, train=True):
    ds = full_train if train else full_test
    return DataLoader(Subset(ds, idxs), batch_size, shuffle=train)

# ----------------------------
# 4) CL Loss Helpers
# ----------------------------
def compute_fisher(model, loader):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n,p in model.named_parameters()}
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        model.zero_grad()
        F.cross_entropy(model(x),y).backward()
        for n,p in model.named_parameters():
            fisher[n] += p.grad.data.pow(2)
    for n in fisher: fisher[n] /= len(loader.dataset)
    model.train()
    return fisher

def ewc_penalty(model, oldp, fisher):
    loss=0
    for n,p in model.named_parameters():
        loss += (fisher[n]*(p-oldp[n]).pow(2)).sum()
    return beta_ewc * loss

def lwf_penalty(new_model, old_model, x):
    if old_model is None: return 0
    old_logits = old_model(x.to(device))
    new_logits = new_model(x.to(device))
    k = old_logits.size(1)
    return alpha_kd * F.kl_div(
        F.log_softmax(new_logits[:,:k]/2,1),
        F.softmax(old_logits/2,1),
        reduction='batchmean'
    )

def average_weights(ws):
    avg = copy.deepcopy(ws[0])
    N = len(ws)
    for k in avg:
        if avg[k].dtype.is_floating_point:
            for w in ws[1:]: avg[k] += w[k]
            avg[k] /= N
    return avg

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            preds = model(x.to(device)).argmax(1)
            correct += (preds == y.to(device)).sum().item()
            total += y.size(0)
    return correct / total

# ----------------------------
# 5) Experiment with metrics
# ----------------------------
def run_experiment(split='iid'):
    client_idxs = create_iid_splits() if split=='iid' else create_noniid_splits(dirichlet_alpha)
    methods = ['ewc','lwf','si']
    results = {}
    for method in methods:
        print(f"\n=== Method: {method.upper()} | Split={split} ===")
        states = [{'fisher':None,'old_params':{},'old_model':None} for _ in range(num_clients)]
        global_model = None
        method_res = {'global_accs':[], 'forget_rates':[]}
        for inc in range(num_increments):
            seen = sum(tasks[:inc+1], [])
            if global_model:
                old_loader = get_loader(
                    [i for i in range(len(full_test.targets)) if full_test.targets[i] in sum(tasks[:inc], [])],
                    False
                )
                pre_acc = evaluate(global_model, old_loader)
            else:
                pre_acc = 0.0
            # Initialize or expand model
            if global_model is None:
                global_model = ResNet32(len(seen))
            else:
                global_model.fc = nn.Linear(global_model.fc.in_features, len(seen)).to(device)
            # Compute EWC/LwF stats
            train_loader_all = get_loader(
                [i for i in range(len(full_train.targets)) if full_train.targets[i] in seen],
                True
            )
            if method == 'ewc':
                Fm = compute_fisher(global_model, train_loader_all)
                OP = {n: p.clone() for n,p in global_model.named_parameters()}
                for st in states:
                    st['fisher'], st['old_params'] = Fm, OP
            if method == 'lwf':
                for st in states:
                    st['old_model'] = copy.deepcopy(global_model).eval()
            # Federated rounds
            round_accs = []
            for _ in range(federated_rounds):
                client_ws = []
                for cid, idxs in enumerate(client_idxs):
                    my_idxs = [i for i in idxs if full_train.targets[i] in seen]
                    if not my_idxs:
                        continue
                    loader = DataLoader(Subset(full_train, my_idxs), batch_size, shuffle=True)
                    lm = copy.deepcopy(global_model)
                    opt = optim.SGD(lm.parameters(), lr=lr, momentum=0.9)
                    for _ in range(local_epochs):
                        for x,y in loader:
                            opt.zero_grad()
                            loss = F.cross_entropy(lm(x.to(device)), y.to(device))
                            if method == 'ewc':
                                loss += ewc_penalty(lm, states[cid]['old_params'], states[cid]['fisher'])
                            elif method == 'lwf':
                                loss += lwf_penalty(lm, states[cid]['old_model'], x)
                            loss.backward()
                            opt.step()
                    client_ws.append(lm.state_dict())
                if client_ws:
                    global_model.load_state_dict(average_weights(client_ws))
                test_loader = get_loader(
                    [i for i in range(len(full_test.targets)) if full_test.targets[i] in seen],
                    False
                )
                gacc = evaluate(global_model, test_loader)
                round_accs.append(gacc)
            # Post-forgetting
            if global_model and inc > 0:
                post_acc = evaluate(global_model, old_loader)
                forget = (pre_acc - post_acc) * 100
            else:
                forget = 0.0
            method_res['global_accs'].append(round_accs)
            method_res['forget_rates'].append(forget)
            print(f"Inc {inc+1}/{num_increments}: rounds accs={np.round(round_accs,3)}, forget={forget:.2f}%")
        results[method] = method_res
    return results

if __name__ == "__main__":
    iid_res = run_experiment('iid')
    noniid_res = run_experiment('noniid')
    print("\nIID Results:", iid_res)
    print("Non-IID Results:", noniid_res)
