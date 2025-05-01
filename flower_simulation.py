import argparse
import copy
from typing import Dict, List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ----------------------------
# 1) Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser(description="CIL-FL Simulation in Flower")
parser.add_argument(
    "--agg_method",
    type=str,
    choices=["fedavg", "fedprox", "scaffold", "glfc"],
    default="fedavg",
    help="Federated aggregation method",
)
parser.add_argument(
    "--cl_method",
    type=str,
    choices=["ewc", "lwf", "si", "none"],
    default="none",
    help="Continual-learning module",
)
parser.add_argument(
    "--num_rounds",
    type=int,
    default=5,
    help="Number of federated rounds per increment",
)
parser.add_argument(
    "--proxmux",
    type=float,
    default=0.1,
    help="Mu parameter for FedProx",
)
args = parser.parse_args()

# ----------------------------
# 2) Hyperparameters & Setup
# ----------------------------
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

# ----------------------------
# 3) Model Definition
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
                nn.BatchNorm2d(out_c),
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

def ResNet32(num_classes: int) -> nn.Module:
    return ResNet(BasicBlock, [5, 5, 5], num_classes).to(device)

# ----------------------------
# 4) Data & Splits
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409),
                         (0.2009, 0.1984, 0.2023)),
])
full_train = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
full_test = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

def create_iid_splits():
    idxs = np.random.permutation(len(full_train))
    sz = len(idxs) // num_clients
    return [idxs[i*sz:(i+1)*sz].tolist() for i in range(num_clients)]

def create_noniid_splits(alpha: float):
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

def get_loader(idxs: List[int], train: bool = True) -> DataLoader:
    ds = full_train if train else full_test
    subset = Subset(ds, idxs)
    return DataLoader(subset, batch_size, shuffle=train)

# ----------------------------
# 5) CL Helpers
# ----------------------------
def compute_fisher(model: nn.Module, loader: DataLoader) -> Dict[str, torch.Tensor]:
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        F.cross_entropy(model(x), y).backward()
        for n, p in model.named_parameters():
            fisher[n] += p.grad.data.pow(2)
    for n in fisher:
        fisher[n] /= len(loader.dataset)
    return fisher

def ewc_penalty(model: nn.Module, old_params, fisher) -> torch.Tensor:
    loss = 0.0
    for n, p in model.named_parameters():
        loss += (fisher[n] * (p - old_params[n]).pow(2)).sum()
    return beta_ewc * loss

def lwf_penalty(model: nn.Module, old_model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    old_logits = old_model(x)
    new_logits = model(x)
    k = old_logits.size(1)
    return alpha_kd * F.kl_div(
        F.log_softmax(new_logits[:, :k]/2, dim=1),
        F.softmax(old_logits/2, dim=1),
        reduction="batchmean",
    )

# ----------------------------
# 6) Client Definition
# ----------------------------
class CILClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_idxs: List[int],
        test_idxs: List[int],
        seen_idxs: List[int],
        cl_method: str,
    ):
        self.cid = cid
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs
        self.seen_idxs = seen_idxs
        self.cl_method = cl_method
        self.model = ResNet32(len(seen_idxs))
        self.old_model = None
        self.fisher = None
        self.old_params = None

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Load global parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

        train_loader = get_loader(self.train_idxs, train=True)

        if self.cl_method == "ewc":
            self.fisher = compute_fisher(self.model, train_loader)
            self.old_params = {n: p.clone() for n, p in self.model.named_parameters()}
        if self.cl_method == "lwf":
            self.old_model = copy.deepcopy(self.model).eval()

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        for _ in range(local_epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                if self.cl_method == "ewc":
                    loss += ewc_penalty(self.model, self.old_params, self.fisher)
                elif self.cl_method == "lwf":
                    loss += lwf_penalty(self.model, self.old_model, x)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(self.train_idxs), {}

    def evaluate(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

        test_loader = get_loader(self.test_idxs, train=False)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                pred = self.model(x.to(device)).argmax(1)
                correct += (pred == y.to(device)).sum().item()
                total += y.size(0)
        loss = float(F.cross_entropy(self.model(x.to(device)), y.to(device)))
        return loss, total, {"accuracy": correct / total}

# ----------------------------
# 7) Simulation Loop
# ----------------------------
if __name__ == "__main__":
    results = []
    client_splits = create_iid_splits() if args.agg_method != "noniid" else create_noniid_splits(dirichlet_alpha)
    for inc in range(num_increments):
        seen = sum(tasks[: inc + 1], [])
        train_idxs = [[i for i in idxs if full_train.targets[i] in seen] for idxs in client_splits]
        test_idxs = [i for i, t in enumerate(full_test.targets) if t in seen]

        # Strategy selection
        if args.agg_method == "fedavg":
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=1.0,
                fraction_eval=1.0,
                min_fit_clients=num_clients,
                min_eval_clients=num_clients,
                min_available_clients=num_clients,
            )
        elif args.agg_method == "fedprox":
            strategy = fl.server.strategy.FedProx(
                mu=args.proxmux,
                fraction_fit=1.0,
                fraction_eval=1.0,
                min_fit_clients=num_clients,
                min_eval_clients=num_clients,
                min_available_clients=num_clients,
            )
        elif args.agg_method == "scaffold":
            from flwr.server.strategy import SCAFFOLD
            strategy = SCAFFOLD(
                fraction_fit=1.0,
                fraction_eval=1.0,
                min_fit_clients=num_clients,
                min_eval_clients=num_clients,
                min_available_clients=num_clients,
            )
        else:
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=1.0,
                fraction_eval=1.0,
                min_fit_clients=num_clients,
                min_eval_clients=num_clients,
                min_available_clients=num_clients,
            )

        def client_fn(cid: str) -> CILClient:
            idx = int(cid)
            return CILClient(
                cid=idx,
                train_idxs=train_idxs[idx],
                test_idxs=test_idxs,
                seen_idxs=seen,
                cl_method=args.cl_method,
            )

        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
        )

        print(f"Increment {inc+1} accuracy: {hist.metrics_centralized['accuracy']}")
        results.append(hist.metrics_centralized["accuracy"])

    print("Final per-increment accuracies:", results)
