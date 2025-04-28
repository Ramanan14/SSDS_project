import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np

# ----------------------------
# 1) Hyperparameters & Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_clients     = 5
num_increments  = 5           # 100 classes → 5 increments of 20
classes_per_inc = 100 // num_increments
central_epochs  = 50          # train upper-bound for 50 epochs on seen classes
local_epochs    = 5           # train lower-bound for 5 epochs on new classes
batch_size      = 128
lr              = 0.01

# ----------------------------
# 2) Transforms & Dataset Load
# ----------------------------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

# assume CIFAR-100 is already downloaded under ./data
full_train = torchvision.datasets.CIFAR100(
    root="./data", train=True,  download=False, transform=transform
)
full_test = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=False, transform=transform
)

# ----------------------------
# 3) IID split of train indices among clients
# ----------------------------
all_idxs   = np.random.permutation(len(full_train))
shard_size = len(all_idxs) // num_clients
client_indices = [
    all_idxs[i*shard_size:(i+1)*shard_size]
    for i in range(num_clients)
]

# ----------------------------
# 4) Class-increment tasks
# ----------------------------
all_classes = list(range(100))
tasks = [
    all_classes[i*classes_per_inc:(i+1)*classes_per_inc]
    for i in range(num_increments)
]

# ----------------------------
# 5) ResNet-32 Definition
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
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(3,16)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32,  layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64,  layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc      = nn.Linear(64*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        return self.fc(out)

def ResNet32(num_classes):
    # 6n+2 layers with n=5 → [5,5,5]
    return ResNet(BasicBlock, [5,5,5], num_classes)

# ----------------------------
# 6) Train/Eval Helpers
# ----------------------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
    return total_loss/len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds==y).sum().item()
            total   += y.size(0)
    return correct/total

# ----------------------------
# 7) Main class-increment loop
# ----------------------------
for t in range(num_increments):
    seen = sum(tasks[:t+1], [])    # all classes up to this increment
    new  = tasks[t]                # newly arrived classes
    print(f"\n=== Increment {t+1}/{num_increments}: seen={seen}, new={new} ===")

    # — Upper-bound: centralized on ALL seen classes —
    ctr_train_idx = [i for i,lbl in enumerate(full_train.targets) if lbl in seen]
    ctr_train_ds  = Subset(full_train, ctr_train_idx)
    ctr_loader    = DataLoader(ctr_train_ds, batch_size, shuffle=True,  num_workers=2)

    ctr_test_idx  = [i for i,lbl in enumerate(full_test.targets)  if lbl in seen]
    ctr_test_ds   = Subset(full_test, ctr_test_idx)
    ctr_test_ld   = DataLoader(ctr_test_ds, batch_size, shuffle=False, num_workers=2)

    model_c = ResNet32(num_classes=len(seen)).to(device)
    opt_c   = optim.SGD(model_c.parameters(), lr=lr, momentum=0.9)
    crit    = nn.CrossEntropyLoss()
    for _ in range(central_epochs):
        train_epoch(model_c, ctr_loader, crit, opt_c)
    acc_c = evaluate(model_c, ctr_test_ld)
    print(f"[Centralized upper-bound] Acc on seen classes = {acc_c:.4f}")

    # — Lower-bound: local-only on NEW classes only —
    local_accs = []
    for cid, idxs in enumerate(client_indices, start=1):
        # client's examples of the new classes only
        my_idxs = [i for i in idxs if full_train.targets[i] in new]
        if not my_idxs:
            print(f" Client {cid:02d}: no new-class data → skip")
            continue

        local_ds = Subset(full_train, my_idxs)
        local_ld = DataLoader(local_ds, batch_size, shuffle=True, num_workers=1)

        model_l = ResNet32(num_classes=len(seen)).to(device)
        opt_l   = optim.SGD(model_l.parameters(), lr=lr, momentum=0.9)
        for _ in range(local_epochs):
            train_epoch(model_l, local_ld, crit, opt_l)
        # evaluate on ALL seen classes to measure forgetting
        acc_l = evaluate(model_l, ctr_test_ld)
        local_accs.append(acc_l)
        print(f" Client {cid:02d} lower-bound Acc (trained on new only) = {acc_l:.4f}")

    if local_accs:
        print(f"[Local lower-bound avg] Acc = {np.mean(local_accs):.4f}")
    else:
        print("[Local lower-bound] No client saw the new classes")
