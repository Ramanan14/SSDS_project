import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients     = 5
num_increments  = 5    # 100 classes → 5 increments of 20
classes_per_inc = 100 // num_increments
local_epochs    = 5
batch_size      = 64
lr              = 0.01
dirichlet_alpha = 0.5
alpha_kd        = 1.0
beta_ewc        = 100.0

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
                nn.Conv2d(in_c,out_c,1,stride,bias=False),
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
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); x = torch.flatten(x,1)
        return self.fc(x)

def ResNet32(num_classes):
    return ResNet(BasicBlock, [5,5,5], num_classes)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4865,0.4409),
                         (0.2009,0.1984,0.2023)),
])
full_train = datasets.CIFAR100("./data", train=True,  download=True, transform=transform)
full_test  = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

def create_iid_splits():
    idxs = np.random.permutation(len(full_train))
    sz   = len(idxs)//num_clients
    return [idxs[i*sz:(i+1)*sz].tolist() for i in range(num_clients)]

def create_noniid_splits(alpha):
    labels = np.array(full_train.targets)
    class_idxs = {c: np.where(labels==c)[0] for c in range(100)}
    client_idxs = [[] for _ in range(num_clients)]
    for idx_list in class_idxs.values():
        props    = np.random.dirichlet([alpha]*num_clients)
        counts   = (props * len(idx_list)).astype(int)
        counts[-1] = len(idx_list)-counts[:-1].sum()
        shuffled = np.random.permutation(idx_list); pos = 0
        for i,c in enumerate(counts):
            client_idxs[i].extend(shuffled[pos:pos+c]); pos += c
    return client_idxs

all_classes = list(range(100))
tasks       = [all_classes[i*classes_per_inc:(i+1)*classes_per_inc]
               for i in range(num_increments)]

def get_loader(classes, train=True):
    ds   = full_train if train else full_test
    idxs = [i for i,l in enumerate(ds.targets) if l in classes]
    return DataLoader(Subset(ds,idxs), batch_size, shuffle=train, num_workers=2)

def compute_fisher(model, loader):
    model.eval()
    fisher = {n:torch.zeros_like(p) for n,p in model.named_parameters()}
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        model.zero_grad()
        F.cross_entropy(model(x), y).backward()
        for n,p in model.named_parameters():
            fisher[n] += p.grad.data.pow(2)
    N = len(loader.dataset)
    return {n:fi/N for n,fi in fisher.items()}

def ewc_loss(model, old_params, fisher):
    loss = 0.0
    for n,p in model.named_parameters():
        if n in old_params and n in fisher and fisher[n].shape==p.shape:
            loss += (fisher[n] * (p-old_params[n]).pow(2)).sum()
    return beta_ewc * loss

def lwf_loss(new_model, old_model, x):
    if old_model is None:
        return 0.0
    with torch.no_grad():
        old_logits = old_model(x.to(device))
    new_logits = new_model(x.to(device))
    k = old_logits.shape[1]
    return alpha_kd * F.kl_div(
        F.log_softmax(new_logits[:,:k]/2, dim=1),
        F.softmax(old_logits/2,    dim=1),
        reduction='batchmean'
    )

def si_loss(model, si_state):
    loss = 0.0
    for n,p in model.named_parameters():
        if n in si_state['omega'] and si_state['omega'][n].shape==p.shape:
            loss += (si_state['omega'][n] * (p - si_state['old_params'][n]).pow(2)).sum()
    return loss

def average_weights(ws):
    avg = copy.deepcopy(ws[0])
    for k in avg:
        if avg[k].dtype.is_floating_point:
            for w in ws[1:]:
                avg[k] += w[k]
            avg[k] /= len(ws)
    return avg

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            pred = model(x.to(device)).argmax(1)
            correct += (pred==y.to(device)).sum().item()
            total += y.size(0)
    return correct/total

def run_scaffold(method, split='iid'):
    client_idxs = create_iid_splits() if split=='iid' else create_noniid_splits(dirichlet_alpha)
    states      = [{'fisher':None,'old_params':{},'old_model':None,
                    'si_state':{'omega':{},'old_params':{}}}
                   for _ in range(num_clients)]
    global_model = None
    results      = []

    for inc in range(num_increments):
        seen = sum(tasks[:inc+1], [])
        new  = tasks[inc]
        print(f"\n-- Increment {inc+1}: new={new} --")

        if global_model is None:
            global_model = ResNet32(len(seen)).to(device)
        else:
            old_fc = global_model.fc
            new_fc = nn.Linear(old_fc.in_features, len(seen)).to(device)
            new_fc.weight.data[:old_fc.out_features] = old_fc.weight.data
            new_fc.bias.data[:old_fc.out_features]   = old_fc.bias.data
            global_model.fc = new_fc

        seen_loader = get_loader(seen, True)
        if method=='ewc':
            Fm = compute_fisher(global_model, seen_loader)
            OP = {n:p.clone() for n,p in global_model.named_parameters()}
            for st in states:
                st['fisher'], st['old_params'] = Fm, OP
        if method=='lwf':
            for st in states:
                st['old_model'] = copy.deepcopy(global_model)
        if method=='si':
            for st in states:
                OP = {n:p.clone() for n,p in global_model.named_parameters()}
                OM = {n:torch.zeros_like(p) for n,p in global_model.named_parameters()}
                st['si_state']['old_params'], st['si_state']['omega'] = OP, OM

        test_loader = get_loader(seen, False)
        C  = {n:torch.zeros_like(p, device=device) for n,p in global_model.named_parameters()}
        Ci = [{n:torch.zeros_like(p, device=device) for n,p in global_model.named_parameters()}
              for _ in range(num_clients)]

        local_states = []
        for cid, idxs in enumerate(client_idxs):
            idxs_new = [i for i in idxs if full_train.targets[i] in new]
            if not idxs_new:
                print(f" client {cid}: no new data")
                continue

            loader = DataLoader(Subset(full_train, idxs_new),
                                batch_size, shuffle=True)
            local = copy.deepcopy(global_model).to(device)
            opt   = optim.SGD(local.parameters(), lr=lr, momentum=0.9)
            initW = {n:p.clone().to(device) for n,p in global_model.named_parameters()}
            st    = states[cid]

            for _ in range(local_epochs):
                for x,y in loader:
                    x,y = x.to(device), y.to(device)
                    opt.zero_grad()
                    out  = local(x)
                    loss = F.cross_entropy(out, y)
                    if method=='ewc':
                        loss += ewc_loss(local, st['old_params'], st['fisher'])
                    elif method=='lwf':
                        loss += lwf_loss(local, st['old_model'], x)
                    elif method=='si':
                        loss += si_loss(local, st['si_state'])

                    loss.backward(retain_graph=True)   # <<== here
                    for n,p in local.named_parameters():
                        if p.grad is not None:
                            p.grad.data += C[n] - Ci[cid][n]
                    opt.step()

                    if method=='si':
                        for n,p in local.named_parameters():
                            oldp  = st['si_state']['old_params'][n]
                            delta = p.detach() - oldp
                            st['si_state']['omega'][n] += p.grad.detach()*delta
                            st['si_state']['old_params'][n] = p.detach().clone()

            local_states.append(local.state_dict())
            for n,p in local.named_parameters():
                Ci[cid][n] = Ci[cid][n] - C[n] + (initW[n] - p).div(lr*local_epochs)

            acc = evaluate(local, test_loader)
            print(f"{method.upper()} client {cid} acc: {acc:.4f}")

        if local_states:
            global_model.load_state_dict(average_weights(local_states))
            for n in C:
                C[n] += sum(Ci[i][n] - C[n] for i in range(num_clients)) / num_clients

        gacc = evaluate(global_model, test_loader)
        print(f"{method.upper()} global acc: {gacc:.4f}")
        results.append(gacc)

    return results

if __name__=="__main__":
    for m in ['ewc','lwf','si']:
        print(f"IID SCAFFOLD + {m.upper()}:", run_scaffold(m,'iid'))
        print(f"Non-IID SCAFFOLD + {m.upper()}:", run_scaffold(m,'noniid'))

