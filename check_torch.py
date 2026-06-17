import torch
import torchvision

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('torchvision has ops:', hasattr(torchvision, 'ops'))
print('torchvision.ops has nms:', hasattr(torchvision.ops, 'nms'))

boxes = torch.tensor([[0,0,10,10],[1,1,11,11]], dtype=torch.float32)
scores = torch.tensor([0.9,0.8])
print('NMS test result:', torchvision.ops.nms(boxes, scores, 0.5))
print('Torch setup is correct ✅')
