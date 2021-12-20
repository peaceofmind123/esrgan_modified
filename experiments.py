from model import Generator
import torch
model      = Generator().cpu()
model_path = f"weights/RRDB_ESRGAN_x4.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model_dict = model.state_dict()
print(len(state_dict.keys()))
print(len(model_dict.keys()))

pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
print(len(pretrained_dict.keys()))

model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict)
print(model)




