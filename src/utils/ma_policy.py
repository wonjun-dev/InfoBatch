# Map-style 데이터셋에서 Custom sampler 적용
from typing import Tuple
import torch
from torchvision import datasets
from torch.utils.data import Sampler
from collections import deque

class MaPruningPolicy(Sampler):
    def __init__(self, data_size, total_epoch, prob=0.5, anneal=0.875):
        self.data_size = data_size
        self.prob = prob
        self.anneal = int(total_epoch * anneal)
        self.threshold = 1.
        self.scores = torch.ones(data_size)
        self.pruned = torch.zeros(data_size, dtype=torch.bool)    # True -> Prune, False -> 사용
        self.data_idxs = torch.arange(data_size)
        self.use_idxs = self.data_idxs
        self.scaler = torch.ones(len(self.use_idxs))
        self.threshold_hist = deque(maxlen=5)
        self.threshold_hist.append(1.)
    
    def update_scores(self, idxs, scores):
        # 매 이터레이션 마다, score 업데이트
        self.scores[idxs] = scores

    def update_policy(self, curr_epoch):
        # 매 에폭 종료시 업데이트
        if curr_epoch < self.anneal:
            self._update_threshold()
            self._update_pruned_idxs()
            self._shuffle()
        else:
            self.use_idxs = self.data_idxs
            self.scaler = torch.ones(len(self.use_idxs))
            self.pruned = torch.zeros(self.data_size, dtype=torch.bool)
            self._shuffle()

    def _update_threshold(self):
        # 매 에폭 이후에, threshold 조정
        # 이전 에폭에 사용했던 subset 만을 이용해서 계산
        subset_scores = self.scores[~self.pruned]
        # self.threshold = torch.mean(subset_scores)
        self.threshold_hist.append(torch.mean(subset_scores))
        self.threshold = sum(self.threshold_hist) / len(self.threshold_hist)

    def _update_pruned_idxs(self):
        # 조정된 threshold를 기준으로 D를 D1, D2로 분할
        # D1에서 soft prunding
        # D1 = D3 + pruned
        # D = D3 + pruned + D2 = D1 + D2
        d2 = self.data_idxs[self.scores >= self.threshold]
        d1 = self.data_idxs[self.scores < self.threshold]
        if len(d1) == 0:
            assert len(d2) == self.data_size
            self.use_idxs = d2
            self.pruned = torch.zeros(self.data_size, dtype=torch.bool)
            self.scaler = torch.ones(self.data_size)
            return

        mask = torch.rand(len(d1)) <= self.prob  # True -> Prune, False -> 사용
        d3 = d1[~mask]
        pruned = d1[mask]
        assert len(d3) + len(pruned) == len(d1)
        assert len(d2) + len(d3) + len(pruned) == self.data_size
        self.use_idxs = torch.cat((d2, d3), dim=0)

        d2_scaler = torch.ones(len(d2))
        d3_scaler = torch.ones(len(d3)) * (1./(1.-self.prob + 1e-6))
        self.scaler = torch.cat((d2_scaler, d3_scaler), dim=0)

        self.pruned[pruned] = True
        self.pruned[d3] = False
        self.pruned[d2] = False

    def _shuffle(self):
        # 셔플링
        perm = torch.randperm(len(self.use_idxs))
        self.use_idxs = self.use_idxs[perm]
        self.scaler = self.scaler[perm]
    
    
    def __iter__(self):
        items = zip(self.use_idxs.tolist(), self.scaler.tolist())
        return iter(items)

    def __len__(self):
        return len(self.use_idxs)


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from dataset import InfoCIFAR10

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = InfoCIFAR10(root='/sources/dataset/cifar10', train=True, download=True, transform=transform)
    policy = MaPruningPolicy(len(dataset))
    policy.update_policy()
   

    train_loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2, sampler=policy)

    for epoch in range(5):
        print(policy.threshold)
        for batch_idx, (x, y, sample_idx, scalers) in enumerate(train_loader):
            scores = torch.rand(len(sample_idx))
            policy.update_scores(sample_idx, scores)
            print(scalers)
        
        policy.update_policy()