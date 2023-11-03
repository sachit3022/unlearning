
class Attack:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    def __call__(self,og,ul):
        #num_models X labels X 2

        raise NotImplementedError