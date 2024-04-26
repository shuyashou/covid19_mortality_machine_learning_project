import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
class RandomForestModel(nn.Module):
    def __init__(self, n_estimators=100):
        super(RandomForestModel, self).__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    def forward(self, x):
        x = x.cpu().numpy()  
        return torch.from_numpy(self.model.predict_proba(x)[:, 1]).float().to(device)

    def predict(self, x):
        if len(x.shape) == 1:  
            x = x.unsqueeze(0) 
        
        x = x.cpu().numpy()
        probabilities = self.model.predict_proba(x)[:, 1]
        return torch.from_numpy(probabilities).float().item()