

class Trainer():
    def __init__(self, cfg):
        self.model = load_model(cfg)
        self.model.lm.train()
        self.model.lm = self.model.lm.float()

    def train(self, train_loader):
