import torch
import train
import test
import saliency_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

print("Training started...")
train.train(device)

print("Testing started...")
test.test(device)

print("Saliency started...")
saliency_map.generate(device)

