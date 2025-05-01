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
mu              = 0.1   # FedProx coefficient (still used in control variate term)

# ----------------------------
# 2) Model (ResNet-32)
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
        for s in [stride] + [1]*(num_blocks-1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x); x = torch.flatten(x,1)
        return self.fc(x)

def ResNet32(num_classes):
    return ResNet(BasicBlock, [5,5,5], num_classes).to(device)

# ----------------------------
# 3) Data & Class Tasks
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
# 4) CL Loss Helpers
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
        if n in old_params:
            loss += (fisher[n] * (p-old_params[n]).pow(2)).sum()
    return beta_ewc * loss

def lwf_loss(new_model, old_model, x):
    if old_model is None: return 0.0
    old_logits = old_model(x.to(device))
    new_logits = new_model(x.to(device))
    k = old_logits.size(1)
    return alpha_kd * F.kl_div(
        F.log_softmax(new_logits[:,:k]/2, dim=1),
        F.softmax(old_logits/2,    dim=1),
        reduction='batchmean'
    )

def si_loss(model, si_state):
    loss = 0.0
    for n,p in model.named_parameters():
        ω = si_state['omega'].get(n)
        if ω is not None:
            oldp = si_state['old_params'][n]
            loss += (ω * (p-oldp.detach()).pow(2)).sum()
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
    c=t=0
    with torch.no_grad():
        for x,y in loader:
            preds = model(x.to(device)).argmax(1)
            c += (preds==y.to(device)).sum().item()
            t += y.size(0)
    return c/t

# ----------------------------
# 5) Precompute Centralized UBs
# ----------------------------
ub_accs = []
for inc in range(num_increments):
    seen = sum(tasks[:inc+1], [])
    central = ResNet32(len(seen))
    opt = optim.SGD(central.parameters(), lr=lr, momentum=0.9)
    loader_tr = get_loader_for_classes(seen, train=True)
    loader_te = get_loader_for_classes(seen, train=False)
    for _ in range(50):
        for x,y in loader_tr:
            opt.zero_grad()
            loss = F.cross_entropy(central(x.to(device)), y.to(device))
            loss.backward(); opt.step()
    ub = evaluate(central, loader_te)
    ub_accs.append(ub)
    print(f"[UB] Increment {inc+1}/{num_increments}: classes {seen} → Acc = {ub:.4f}")

# ----------------------------
# 6) Modified SCAFFOLD + CL w/ Metrics
# ----------------------------
def run_scaffold_with_metrics(method, split='iid'):
    client_idxs = (create_iid_splits() if split=='iid'
                   else create_noniid_splits(dirichlet_alpha))
    # Per-client CL states
    states = [{'fisher':None,'old_params':{},'old_model':None,
               'si_state':{'omega':{},'old_params':{}}}
              for _ in range(num_clients)]

    # containers for metrics
    metrics = {
        'round_accs': [],         # list of lists (per-inc: per-round accs)
        'to_80': [],              # rounds to hit 0.8·UB
        'forgetting': []          # avg forgetting per increment
    }

    global_model = None
    prev_final_accs = []  # keep final accs per increment for forgetting

    for inc in range(num_increments):
        seen = sum(tasks[:inc+1], [])
        new  = tasks[inc]
        print(f"\n=== Method={method.upper()} | Split={split} | Increment {inc+1}/{num_increments}: new={new} ===")

        # (re)initialize / expand global model
        if global_model is None:
            global_model = ResNet32(len(seen))
        else:
            global_model.fc = nn.Linear(global_model.fc.in_features, len(seen)).to(device)

        # update CL states based on global_model
        loader_cl = get_loader_for_classes(seen, train=True)
        if method=='ewc':
            Fm = compute_fisher(global_model, loader_cl)
            OP = {n:p.clone() for n,p in global_model.named_parameters()}
            for st in states:
                st['fisher'], st['old_params'] = Fm, OP
        if method=='lwf':
            for st in states:
                st['old_model'] = copy.deepcopy(global_model).eval()
        if method=='si':
            for st in states:
                OP = {n:p.clone() for n,p in global_model.named_parameters()}
                OM = {n:st['si_state']['omega'].get(n, torch.zeros_like(p))
                      for n,p in global_model.named_parameters()}
                st['si_state']['old_params'], st['si_state']['omega'] = OP, OM

        # prepare SCAFFOLD variates
        C  = {n:torch.zeros_like(p, device=device) for n,p in global_model.named_parameters()}
        Ci = [{n:torch.zeros_like(p, device=device) for n,p in global_model.named_parameters()}
              for _ in range(num_clients)]

        # testing loader for seen classes
        test_loader = get_loader_for_classes(seen, train=False)
        threshold = 0.8 * ub_accs[inc]

        round_accs = []
        hit_round = None

        # --- Run 15 federated rounds ---
        for rnd in range(1, federated_rounds+1):
            local_states = []
            for cid, idxs in enumerate(client_idxs):
                # filter this client's new data
                my_idxs = [i for i in idxs if full_train.targets[i] in new]
                if not my_idxs:
                    continue

                loader = DataLoader(Subset(full_train, my_idxs),
                                    batch_size, shuffle=True)
                local = copy.deepcopy(global_model).to(device)
                opt   = optim.SGD(local.parameters(), lr=lr, momentum=0.9)
                st    = states[cid]
                initW = {n:p.clone().to(device) for n,p in global_model.named_parameters()}

                for _ in range(local_epochs):
                    for x,y in loader:
                        x,y = x.to(device), y.to(device)
                        opt.zero_grad()
                        out = local(x)

                        loss = F.cross_entropy(out, y)
                        if method=='ewc':
                            loss += ewc_loss(local, st['old_params'], st['fisher'])
                        elif method=='lwf':
                            loss += lwf_loss(local, st['old_model'], x)
                        elif method=='si':
                            loss += si_loss(local, st['si_state'])

                        # FedProx term (same for SCAFFOLD)
                        prox = sum((lp-gp).pow(2).sum()
                                   for (_,lp),(_,gp) in zip(local.named_parameters(),
                                                           global_model.named_parameters()))
                        loss += (mu/2)*prox

                        # SCAFFOLD control variate adjustment
                        loss.backward(retain_graph=(method=='si'))
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
                # update Ci[cid]
                for n,p in local.named_parameters():
                    Ci[cid][n] = Ci[cid][n] - C[n] + (initW[n] - p).div(lr*local_epochs)

            # aggregate global
            if local_states:
                global_model.load_state_dict(average_weights(local_states))
                for n in C:
                    C[n] += sum(Ci[i][n] - C[n] for i in range(num_clients)) / num_clients

            # evaluate after this round
            gacc = evaluate(global_model, test_loader)
            round_accs.append(gacc)
            print(f" Round {rnd:02d}/{federated_rounds}: Global Acc = {gacc:.4f}")

            # record first hit ≥80%UB
            if hit_round is None and gacc >= threshold:
                hit_round = rnd

        # metrics for this increment
        metrics['round_accs'].append(round_accs)
        metrics['to_80'].append(hit_round or federated_rounds)

        # compute forgetting: compare each prior increment’s final acc vs now
        if inc==0:
            metrics['forgetting'].append(0.0)
        else:
            drops = []
            for j in range(inc):
                # eval on classes of task j
                loader_j = get_loader_for_classes(tasks[j], train=False)
                acc_now  = evaluate(global_model, loader_j)
                acc_then = prev_final_accs[j]
                drops.append(acc_then - acc_now)
            metrics['forgetting'].append(sum(drops)/len(drops))

        # save final acc of this increment for future forgetting
        prev_final_accs.append(round_accs[-1])

    return metrics

if __name__=="__main__":
    for method in ['ewc','lwf','si']:
        print(f"\n##### SCAFFOLD + {method.upper()} (IID) #####")
        m_iid = run_scaffold_with_metrics(method,'iid')
        print("\n##### SCAFFOLD + {method.upper()} (Non-IID) #####")
        m_noniid = run_scaffold_with_metrics(method,'noniid')
