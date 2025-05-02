import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy

# ----------------------------
# 1) Setup & Hyperparameters
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients      = 5
num_increments   = 5
classes_per_inc  = 100 // num_increments
local_epochs     = 5
federated_rounds = 15       # ← 15 rounds per increment
batch_size       = 64
lr               = 0.01
mu               = 0.1     # FedProx μ
dirichlet_alpha  = 0.5
alpha_kd         = 1.0
beta_ewc         = 100.0

# ----------------------------
# 2) Model Definition (ResNet-32)
# ----------------------------
def conv3x3(in_c,out_c,stride=1):
    return nn.Conv2d(in_c,out_c,3,stride,padding=1,bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_c,out_c,stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_c,out_c,stride)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c,out_c)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Identity()
        if stride!=1 or in_c!=out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c,out_c,1,stride,bias=False),
                nn.BatchNorm2d(out_c)
            )
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes):
        super().__init__()
        self.in_planes = 16
        self.conv1     = conv3x3(3,16)
        self.bn1       = nn.BatchNorm2d(16)
        self.layer1    = self._make_layer(block,16,layers[0],1)
        self.layer2    = self._make_layer(block,32,layers[1],2)
        self.layer3    = self._make_layer(block,64,layers[2],2)
        self.avgpool   = nn.AdaptiveAvgPool2d((1,1))
        self.fc        = nn.Linear(64*block.expansion,num_classes)
    def _make_layer(self,block,planes,blocks,stride):
        layers = []
        for s in [stride]+[1]*(blocks-1):
            layers.append(block(self.in_planes,planes,s))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); x = torch.flatten(x,1)
        return self.fc(x)

def ResNet32(num_classes):
    return ResNet(BasicBlock,[5,5,5],num_classes).to(device)

# ----------------------------
# 3) Data & Class-Increment Tasks
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4865,0.4409),(0.2009,0.1984,0.2023)),
])
full_train = datasets.CIFAR100("./data",train=True,download=True,transform=transform)
full_test  = datasets.CIFAR100("./data",train=False,download=True,transform=transform)

all_classes = list(range(100))
tasks = [all_classes[i*classes_per_inc:(i+1)*classes_per_inc]
         for i in range(num_increments)]

def create_iid_splits():
    idxs = np.random.permutation(len(full_train))
    sz   = len(idxs)//num_clients
    return [idxs[i*sz:(i+1)*sz].tolist() for i in range(num_clients)]

def create_noniid_splits(alpha):
    labels = np.array(full_train.targets)
    class_idxs = {c: np.where(labels==c)[0] for c in range(100)}
    client_idxs = [[] for _ in range(num_clients)]
    for idx_list in class_idxs.values():
        props = np.random.dirichlet([alpha]*num_clients)
        counts = (props*len(idx_list)).astype(int)
        counts[-1] = len(idx_list)-counts[:-1].sum()
        shuffled = np.random.permutation(idx_list)
        pos = 0
        for i,cnt in enumerate(counts):
            client_idxs[i].extend(shuffled[pos:pos+cnt])
            pos += cnt
    return client_idxs

def get_loader_for_classes(classes, train=True):
    ds   = full_train if train else full_test
    idxs = [i for i,l in enumerate(ds.targets) if l in classes]
    return DataLoader(Subset(ds,idxs),batch_size,shuffle=train)

# ----------------------------
# 4) CL Loss Helpers
# ----------------------------
def compute_fisher(model, loader):
    model.eval()
    fisher = {n:torch.zeros_like(p) for n,p in model.named_parameters()}
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        model.zero_grad()
        F.cross_entropy(model(x),y).backward()
        for n,p in model.named_parameters():
            fisher[n] += p.grad.data.pow(2)
    N = len(loader.dataset)
    for n in fisher: fisher[n] /= N
    model.train()
    return fisher

def ewc_loss(model, old_params, fisher, lam=beta_ewc):
    loss=0
    for n,p in model.named_parameters():
        if n in old_params:
            loss += (fisher[n]*(p-old_params[n]).pow(2)).sum()
    return lam*loss

def lwf_loss(new_model, old_model, x, T=2.0, alpha=alpha_kd):
    if old_model is None: return 0.0
    old_logits = old_model(x.to(device))
    new_logits = new_model(x.to(device))
    k = old_logits.size(1)
    return alpha*T*T*F.kl_div(
        F.log_softmax(new_logits[:,:k]/T,dim=1),
        F.softmax(old_logits/T,dim=1),
        reduction='batchmean'
    )

def si_loss(model, si_state, c=0.1):
    loss=0
    for n,p in model.named_parameters():
        if n in si_state['omega']:
            loss += (si_state['omega'][n]*(p-si_state['old_params'][n].detach()).pow(2)).sum()
    return c*loss

def average_weights(ws):
    avg = copy.deepcopy(ws[0])
    for k in avg:
        if avg[k].dtype.is_floating_point:
            for w in ws[1:]: avg[k] += w[k]
            avg[k] /= len(ws)
    return avg

def evaluate(model, loader):
    model.eval()
    correct,total=0,0
    with torch.no_grad():
        for x,y in loader:
            preds = model(x.to(device)).argmax(1)
            correct += (preds==y.to(device)).sum().item()
            total   += y.size(0)
    return correct/total

# ----------------------------
# 5) Experiment loop with metrics
# ----------------------------
def run_prox_with_metrics(split='iid'):
    client_idxs = create_iid_splits() if split=='iid' else create_noniid_splits(dirichlet_alpha)
    methods = ['ewc','lwf','si']
    metrics = {m:{
        'global_accs': [],
        'forgetting': []
    } for m in methods}

    # 1) compute centralized upper-bounds per increment
    ub_accs = []
    for inc in range(num_increments):
        seen = sum(tasks[:inc+1],[])
        central = ResNet32(len(seen))
        opt = optim.SGD(central.parameters(),lr=lr,momentum=0.9)
        loader_tr = get_loader_for_classes(seen,train=True)
        loader_te = get_loader_for_classes(seen,train=False)
        for _ in range(50):
            for x,y in loader_tr:
                opt.zero_grad()
                loss = F.cross_entropy(central(x.to(device)),y.to(device))
                loss.backward(); opt.step()
        ub = evaluate(central,loader_te)
        ub_accs.append(ub)
        print(f"UB Increment {inc+1}: {ub:.4f}")

    # 2) federated Prox + CL
    for method in methods:
        print(f"\n=== {method.upper()} + FedProx ({split}) ===")
        states = [{'fisher':None,'old_params':{},'old_model':None,
                   'si_state':{'omega':{},'old_params':{}}}
                  for _ in range(num_clients)]
        global_model = None

        for inc in range(num_increments):
            seen = sum(tasks[:inc+1],[])
            new  = tasks[inc]
            loader_te = get_loader_for_classes(seen,train=False)

            if global_model is None:
                global_model = ResNet32(len(seen))
            else:
                global_model.fc = nn.Linear(global_model.fc.in_features,len(seen)).to(device)

            loader_cl = get_loader_for_classes(seen,train=True)
            if method=='ewc':
                fisher = compute_fisher(global_model,loader_cl)
                oldp   = {n:p.clone() for n,p in global_model.named_parameters()}
                for st in states:
                    st['fisher'],st['old_params']=fisher,oldp
            if method=='lwf':
                for st in states:
                    st['old_model']=copy.deepcopy(global_model).eval()
            if method=='si':
                for st in states:
                    op = {n:p.clone() for n,p in global_model.named_parameters()}
                    om = {n:st['si_state']['omega'].get(n,torch.zeros_like(p))
                          for n,p in global_model.named_parameters()}
                    st['si_state']['old_params'],st['si_state']['omega']=op,om

            print(f"\n-- Increment {inc+1}: new={new} --")
            round_accs = []

            for rnd in range(1, federated_rounds+1):
                local_ws = []
                for cid,idxs in enumerate(client_idxs):
                    my = [i for i in idxs if full_train.targets[i] in new]
                    if not my: continue
                    loader_tr = DataLoader(Subset(full_train,my), batch_size,shuffle=True)
                    local = copy.deepcopy(global_model)
                    opt = optim.SGD(local.parameters(),lr=lr,momentum=0.9)
                    st  = states[cid]

                    for _ in range(local_epochs):
                        for x,y in loader_tr:
                            x,y = x.to(device),y.to(device)
                            opt.zero_grad()
                            out = local(x)
                            loss=F.cross_entropy(out,y)
                            if method=='ewc': loss+=ewc_loss(local,st['old_params'],st['fisher'])
                            if method=='lwf': loss+=lwf_loss(local,st['old_model'],x)
                            if method=='si':  loss+=si_loss(local,st['si_state'])
                            prox=0
                            for (n,lp),(__,gp) in zip(local.named_parameters(),global_model.named_parameters()):
                                prox += (lp-gp).pow(2).sum()
                            loss += (mu/2)*prox
                            if method=='si':
                                loss.backward(retain_graph=True)
                            else:
                                loss.backward()
                            opt.step()
                            if method=='si':
                                for n,p in local.named_parameters():
                                    delta = p.detach()-st['si_state']['old_params'][n]
                                    st['si_state']['omega'][n] += p.grad.detach()*delta
                                    st['si_state']['old_params'][n] = p.detach().clone()

                    local_ws.append(local.state_dict())

                if local_ws:
                    global_model.load_state_dict(average_weights(local_ws))

                gacc = evaluate(global_model,loader_te)
                round_accs.append(gacc)
                print(f" Round {rnd:02d}: Global acc = {gacc:.4f}")

            metrics[method]['global_accs'].append(round_accs)
            if inc == 0:
                metrics[method]['forgetting'].append(0.0)
            else:
                forgets = []
                for j in range(inc):
                    loader_j = get_loader_for_classes(tasks[j],train=False)
                    acc_now  = evaluate(global_model,loader_j)
                    acc_then = metrics[method]['global_accs'][j][-1]
                    forgets.append(acc_then - acc_now)
                metrics[method]['forgetting'].append(sum(forgets)/len(forgets))

    return metrics, ub_accs

if __name__ == "__main__":
    iid_metrics, iid_ub = run_prox_with_metrics('iid')
    noniid_metrics, noniid_ub = run_prox_with_metrics('noniid')

    print("\n\n== Results (IID) ==")
    for m in iid_metrics:
        print(f"{m.upper()}:")
        for inc in range(num_increments):
            print(
                f" Inc {inc+1}: "
                f"Acc@15=[{', '.join(f'{a:.3f}' for a in iid_metrics[m]['global_accs'][inc])}], "
                f"Forget={iid_metrics[m]['forgetting'][inc]:.3f}"
            )

    print("\n\n== Results (Non-IID) ==")
    for m in noniid_metrics:
        print(f"{m.upper()}:")
        for inc in range(num_increments):
            print(
                f" Inc {inc+1}: "
                f"Acc@15=[{', '.join(f'{a:.3f}' for a in noniid_metrics[m]['global_accs'][inc])}], "
                f"Forget={noniid_metrics[m]['forgetting'][inc]:.3f}"
            )
