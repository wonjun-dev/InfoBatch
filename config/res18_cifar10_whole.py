class Config:
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 128
        self.max_lr = 0.1
        self.momentum = 0.9
        self.pct_start = 0.3
        self.weight_decay = 5e-4
        self.label_smooth = 0.
        self.stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
