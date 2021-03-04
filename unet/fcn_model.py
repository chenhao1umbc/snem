# Deep learning modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class InitNet(nn.Module):

    def __init__(self, n_classes):
        super(InitNet, self).__init__()
        self.n_classes = n_classes
        self.fc1 = nn.Linear(128, 128*self.n_classes)
        self.fc2 = nn.Linear(128*self.n_classes, 128*self.n_classes)
        self.fc3 = nn.Linear(128*self.n_classes, 128*self.n_classes)
        self.classifier = nn.Linear(self.n_classes*20*128, n_classes)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        component_output = x.view(-1, self.n_classes, 20, 128)
        flatten_component = component_output.view(-1, self.n_classes*20*128)
        class_output = self.classifier(flatten_component)
        normal_class_output = self.output(class_output)
        return component_output, normal_class_output
