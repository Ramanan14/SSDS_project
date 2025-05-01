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
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], 1)
        self.layer2 = self._make_layer(block, 32, layers[1], 2)
        self.layer3 = self._make_layer(block, 64, layers[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc      = nn.Linear(64*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1]*(num_blocks-1):
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
    return ResNet(BasicBlock, [5,5,5], num_classes)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4865,0.4409),(0.2009,0.1984,0.2023)),
])
full_train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
full_test  = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

def create_iid_splits():
    idxs = np.random.permutation(len(full_train))
    size = len(idxs)//num_clients
    return [idxs[i*size:(i+1)*size].tolist() for i in range(num_clients)]

def create_noniid_splits(alpha):
    labels = np.array(full_train.targets)
    class_idxs = {c: np.where(labels==c)[0] for c in range(100)}
    client_idxs = [[] for _ in range(num_clients)]
    for idx_list in class_idxs.values():
        props = np.random.dirichlet([alpha]*num_clients)
        counts = (props * len(idx_list)).astype(int)
        counts[-1] = len(idx_list)-counts[:-1].sum()
        shuffled = np.random.permutation(idx_list)
        pos = 0
        for i,cnt in enumerate(counts):
            client_idxs[i].extend(shuffled[pos:pos+cnt])
            pos += cnt
    return client_idxs

all_classes = list(range(100))
tasks = [all_classes[i*classes_per_inc:(i+1)*classes_per_inc] for i in range(num_increments)]

def get_loader(classes, train=True):
    dataset = full_train if train else full_test
    idxs = [i for i,l in enumerate(dataset.targets) if l in classes]
    return DataLoader(Subset(dataset, idxs), batch_size, shuffle=train)

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
    for n in fisher:
        fisher[n] /= N
    model.train()
    return fisher

def ewc_loss(model, old_params, fisher, lam=100):
    loss = 0
    for name,p in model.named_parameters():
        if name in old_params and name in fisher and fisher[name].shape==p.shape:
            loss += (fisher[name]*(p-old_params[name]).pow(2)).sum()
    return lam*loss

def lwf_loss(new_model, old_model, x, T=2.0, alpha=1.0):
    if old_model is None:
        return 0
    old_logits = old_model(x.to(device))
    new_logits = new_model(x.to(device))
    k = old_logits.shape[1]
    return alpha*T*T*F.kl_div(
        F.log_softmax(new_logits[:,:k]/T, dim=1),
        F.softmax(old_logits/T,    dim=1),
        reduction='batchmean'
    )

def si_loss(model, si_state, c=0.1):
    loss = 0
    for name,p in model.named_parameters():
        if name in si_state['omega'] and si_state['omega'][name].shape==p.shape:
            loss += (si_state['omega'][name]*(p-si_state['old_params'][name].detach()).pow(2)).sum()
    return c*loss

def average_weights(weights):
    avg = copy.deepcopy(weights[0])
    n = len(weights)
    for k,v in avg.items():
        if v.dtype.is_floating_point:
            for w in weights[1:]:
                avg[k] += w[k]
            avg[k] /= n
    return avg

def evaluate(model, loader):
    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for x,y in loader:
            correct += (model(x.to(device)).argmax(1)==y.to(device)).sum().item()
            total += y.size(0)
    return correct/total

def run_experiment(split='iid'):
    client_idxs = create_iid_splits() if split=='iid' else create_noniid_splits(dirichlet_alpha)
    methods = ['ewc','lwf','si']
    results = {m: [] for m in methods}

    for method in methods:
        states = [{'fisher':None,'old_params':{},'old_model':None,
                   'si_state':{'omega':{},'old_params':{}}}
                  for _ in range(num_clients)]
        global_model = None

        for inc in range(num_increments):
            seen = sum(tasks[:inc+1],[])
            new  = tasks[inc]

            if global_model is None:
                global_model = ResNet32(len(seen)).to(device)
            else:
                global_model.fc = nn.Linear(global_model.fc.in_features,len(seen)).to(device)

            train_loader = get_loader(seen, train=True)
            if method=='ewc':
                fisher = compute_fisher(global_model, train_loader)
                oldp = {n:p.clone() for n,p in global_model.named_parameters()}
                for st in states:
                    st['fisher'], st['old_params'] = fisher, oldp
            if method=='lwf':
                for st in states:
                    st['old_model'] = copy.deepcopy(global_model)
            if method=='si':
                for st in states:
                    oldp = {n:p.clone() for n,p in global_model.named_parameters()}
                    om = {}
                    for n,p in global_model.named_parameters():
                        om[n] = st['si_state']['omega'].get(n, torch.zeros_like(p))
                        if om[n].shape!=p.shape:
                            om[n] = torch.zeros_like(p)
                    st['si_state']['old_params'] = oldp
                    st['si_state']['omega'] = om

            test_loader = get_loader(seen, train=False)
            client_weights = []
            for cid, idxs in enumerate(client_idxs):
                my = [i for i in idxs if full_train.targets[i] in new]
                if not my:
                    continue
                loader = DataLoader(Subset(full_train,my), batch_size, shuffle=True)
                lm = copy.deepcopy(global_model)
                opt = optim.SGD(lm.parameters(), lr=lr, momentum=0.9)
                st = states[cid]

                for _ in range(local_epochs):
                    for x,y in loader:
                        opt.zero_grad()
                        loss = F.cross_entropy(lm(x.to(device)), y.to(device))
                        if method=='ewc':
                            loss += ewc_loss(lm, st['old_params'], st['fisher'])
                        elif method=='lwf':
                            loss += lwf_loss(lm, st['old_model'], x)
                        elif method=='si':
                            loss += si_loss(lm, st['si_state'])
                        if method=='si':
                            loss.backward(retain_graph=True)
                        else:
                            loss.backward()
                        opt.step()
                        if method=='si':
                            for n,p in lm.named_parameters():
                                delta = p.detach()-st['si_state']['old_params'][n]
                                st['si_state']['omega'][n] += p.grad.detach()*delta
                                st['si_state']['old_params'][n] = p.detach().clone()

                print(f"Client {cid} acc: {evaluate(lm,test_loader):.4f}")
                client_weights.append(lm.state_dict())

            if client_weights:
                global_model.load_state_dict(average_weights(client_weights))

            print(f"Global acc: {evaluate(global_model,test_loader):.4f}")
            results[method].append(evaluate(global_model,test_loader))

    return results

if __name__=="__main__":
    print("IID:", run_experiment('iid'))
    print("non-IID:", run_experiment('noniid'))
