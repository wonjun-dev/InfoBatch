class Config:
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 128
        self.max_lr = 0.03
        self.momentum = 0.9
        self.pct_start = 0.3
        self.weight_decay = 5e-4
        self.label_smooth = 0.1
        self.stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
