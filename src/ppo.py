class PPONet:

    def __init__(self):
        pass;

class PPOLearner:

    def __init__(self, params):
        self.ppo_net = PPONet(self.nr_input_features, self.nr_actions).to(self.device)


    def policy(self):
        pass;

    def predict_policy(self):
        pass;

    def predict_value(self):
        pass;

    def update(self):
        pass;