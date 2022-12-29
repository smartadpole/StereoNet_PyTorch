import numpy as np
import torch
from stereonet.model import StereoNet
from stereonet import utils as utils

# Load in the image pair as numpy uint8 arrays
# sample = {'left': utils.image_loader(path_to_left_rgb_image_file),
#           'right': utils.image_loader(path_to_right_rgb_image_file)
#           }

# Here just creating a random image
rng = np.random.default_rng()
sample = {'left': (rng.random((540, 960, 3))*255).astype(np.uint8),  # [height, width, channel],
          'right': (rng.random((540, 960, 3))*255).astype(np.uint8)  # [height, width, channel]
          }

# Transform the single image pair into a torch.Tensor then into a
# batch of shape [batch, channel, height, width]
transformers = [utils.ToTensor(), utils.PadSampleToBatch()]
for transformer in transformers:
    sample = transformer(sample)

# Load in the model from the trained checkpoint
# model = StereoNet.load_from_checkpoint(path_to_checkpoint)

# Here just instantiate the model with random weights
model = StereoNet()
model.load_from_checkpoint("epoch=20-step=744533.ckpt")

# Set the model to eval and run the forward method without tracking gradients
model.eval()
# with torch.no_grad():
#     batched_prediction = model(sample)

onnx_input_L = torch.rand(1, 3, 400, 640)
onnx_input_R = torch.rand(1, 3, 400, 640)
# onnx_input_L = onnx_input_L.to("cuda:0")
# onnx_input_R = onnx_input_R.to("cuda:0")
torch.onnx.export(model,
                  (onnx_input_L, onnx_input_R),
                  "{}.onnx".format("Stereo_ljx"),
                  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=12,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['left', 'right'],  # the model's input names
                  output_names=['output'])

# Remove the batch diemnsion and switch back to channels last notation
# single_prediction = batched_prediction[0].numpy()  # [batch, ...] -> [...]
# single_prediction = np.moveaxis(single_prediction, 0, 2)  # [channel, height, width] -> [height, width, channel]
