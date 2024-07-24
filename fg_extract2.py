# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# import sys

# def process_image(image_path):
#     # Load the image
#     img = cv.imread(image_path)

#     # Convert to HSV color space
#     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#     # Define the lower and upper bounds for skin color in HSV
#     lower = np.array([0, 20, 70], dtype="uint8")
#     upper = np.array([20, 255, 255], dtype="uint8")

#     # Threshold the image to extract only the skin color pixels
#     mask = cv.inRange(hsv, lower, upper)

#     # Perform morphological transformations to remove noise
#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#     mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

#     # Detect contours in the binary image
#     contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     # Sort the contours according to area
#     if contours:
#         contours = sorted(contours, key=cv.contourArea, reverse=True)

#         # Focus on the largest contour (assuming it's the hand or finger)
#         largest_contour = contours[0]

#         # Create a binary mask using the largest contour
#         mask = np.zeros_like(mask)
#         cv.drawContours(mask, [largest_contour], 0, 255, -1)

#         # Perform post-processing to further improve the quality of the segmented region
#         mask = cv.GaussianBlur(mask, (5, 5), 0)

#         # Segment the hand or finger region from the rest of the image
#         hand = cv.bitwise_and(img, img, mask=mask)

#         # Convert to grayscale and apply adaptive thresholding
#         hand_gray = cv.cvtColor(hand, cv.COLOR_BGR2GRAY)
#         hand_thresh = cv.adaptiveThreshold(hand_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

#         # Display the results
#         plt.figure(figsize=(8, 8))
#         plt.subplot(1, 2, 1)
#         plt.title('Segmented Region')
#         plt.imshow(hand_thresh, cmap='gray')
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         plt.title('Original Image with Mask')
#         plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#         plt.axis('off')

#         plt.show()
#     else:
#         print("No contours found!")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python script.py <image_path>")
#         sys.exit(1)

#     image_path = sys.argv[1]
#     process_image(image_path)

# import cv2 as cv
# import numpy as np
# import torch
# from torch.autograd import Variable
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import sys
# import os
# from u2net import U2NET  # Ensure you have the U2NET model file in the same directory

# def load_model(model_path):
#     print("Loading U^2-Net model...")
#     net = U2NET(3, 1)
#     net.load_state_dict(torch.load(model_path, map_location='cpu'))
#     net.eval()
#     return net

# def remove_background(image_path, model):
#     transform = transforms.Compose([
#         transforms.Resize((320, 320)),
#         transforms.ToTensor()
#     ])

#     # Load the image
#     img = Image.open(image_path).convert('RGB')
#     img_transformed = transform(img).unsqueeze(0)
    
#     with torch.no_grad():
#         d1, d2, d3, d4, d5, d6, d7 = model(Variable(img_transformed))
#         pred = d1[:, 0, :, :]
#         pred = norm_pred(pred)
#         pred = pred.squeeze().cpu().data.numpy()

#     mask = cv.resize(pred, (img.width, img.height)) > 0.5
#     result = np.array(img)
#     result[~mask] = [255, 255, 255]  # Set background to white

#     return result

# def norm_pred(d):
#     ma = torch.max(d)
#     mi = torch.min(d)
#     dn = (d - mi) / (ma - mi)
#     return dn

# def process_image(image_path, model_path):
#     model = load_model(model_path)

#     # Remove background
#     img = remove_background(image_path, model)

#     # Save the result
#     result_path = 'result.png'
#     cv.imwrite(result_path, cv.cvtColor(img, cv.COLOR_RGB2BGR))
#     return result_path

# def display_image(image_path):
#     img = cv.imread(image_path)
#     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python script.py <image_path>")
#         sys.exit(1)

#     image_path = sys.argv[1]
#     model_path = 'u2netp.pth'  # Ensure the model file is in the same directory

#     if not os.path.exists(model_path):
#         print(f"Model file not found: {model_path}")
#         sys.exit(1)

#     result_path = process_image(image_path, model_path)
#     display_image(result_path)

# import torch
# from carvekit.api.high import HiInterface

# # Check doc strings for more information
# interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
#                         batch_size_seg=5,
#                         batch_size_matting=1,
#                         device='cuda' if torch.cuda.is_available() else 'cpu',
#                         seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
#                         matting_mask_size=2048,
#                         trimap_prob_threshold=231,
#                         trimap_dilation=30,
#                         trimap_erosion_iters=5,
#                         fp16=False)
# images_without_background = interface(['images/fgnew3.jpeg'])
# cat_wo_bg = images_without_background[0]
# cat_wo_bg.save('images/check.jpeg')


# from rembg import remove
# from PIL import Image
# input_path = input("images/fgnew3.jpeg")
# output_path = input("images/output.jpeg")
# # open_image = input("Open image after when finished? (Y/n): ")
# input_image = Image.open(input_path)
# output_image = remove(input_image)
# output_image.save(output_path)
# print("Background Removed Succesfully !")
# Image.open(output_path)

from rembg import remove
from PIL import Image
import cv2

input_path = 'images/fg1.jpeg'
output_path = 'images/output2.jpeg'

input = cv2.imread(input_path)
output = remove(input)
cv2.imwrite(output_path, output)
Image.open(output_path)


# from backgroundremover.bg import remove
# def remove_bg(src_img_path, out_img_path):
#     model_choices = ["u2net", "u2net_human_seg", "u2netp"]
#     f = open(src_img_path, "rb")
#     data = f.read()
#     img = remove(data, model_name=model_choices[0],
#                  alpha_matting=True,
#                  alpha_matting_foreground_threshold=240,
#                  alpha_matting_background_threshold=10,
#                  alpha_matting_erode_structure_size=10,
#                  alpha_matting_base_size=1000)
#     f.close()
#     f = open(out_img_path, "wb")
#     f.write(img)
#     f.close()
# import io
# import os
# import typing
# from PIL import Image
# from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
# from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
# from pymatting.util.util import stack_images
# from scipy.ndimage.morphology import binary_erosion
# import moviepy.editor as mpy
# import numpy as np
# import torch
# import torch.nn.functional
# import torch.nn.functional
# from hsh.library.hash import Hasher
# from .u2net import detect, u2net
# from . import github

# # closes https://github.com/nadermx/backgroundremover/issues/18
# # closes https://github.com/nadermx/backgroundremover/issues/112
# try:
#     if torch.cuda.is_available():
#         DEVICE = torch.device('cuda:0')
#     elif torch.backends.mps.is_available():
#         DEVICE = torch.device('mps')
#     else:
#         DEVICE = torch.device('cpu')
# except Exception as e:
#     print(f"Using CPU.  Setting Cuda or MPS failed: {e}")
#     DEVICE = torch.device('cpu')

# class Net(torch.nn.Module):
#     def __init__(self, model_name):
#         super(Net, self).__init__()
#         hasher = Hasher()
#         model = {
#             'u2netp': (u2net.U2NETP,
#                        'e4f636406ca4e2af789941e7f139ee2e',
#                        '1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy',
#                        'U2NET_PATH'),
#             'u2net': (u2net.U2NET,
#                       '09fb4e49b7f785c9f855baf94916840a',
#                       '1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
#                       'U2NET_PATH'),
#             'u2net_human_seg': (u2net.U2NET,
#                                 '347c3d51b01528e5c6c071e3cff1cb55',
#                                 '1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P',
#                                 'U2NET_PATH')
#         }[model_name]

#         if model_name == "u2netp":
#             net = u2net.U2NETP(3, 1)
#             path = os.environ.get(
#                 "U2NETP_PATH",
#                 os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
#             )
#             if (
#                 not os.path.exists(path)
#             ):
#                 github.download_files_from_github(
#                     path, model_name
#                 )

#         elif model_name == "u2net":
#             net = u2net.U2NET(3, 1)
#             path = os.environ.get(
#                 "U2NET_PATH",
#                 os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
#             )
#             if (
#                 not os.path.exists(path)
#                 #or hasher.md5(path) != "09fb4e49b7f785c9f855baf94916840a"
#             ):
#                 github.download_files_from_github(
#                     path, model_name
#                 )

#         elif model_name == "u2net_human_seg":
#             net = u2net.U2NET(3, 1)
#             path = os.environ.get(
#                 "U2NET_PATH",
#                 os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
#             )
#             if (
#                 not os.path.exists(path)
#                 #or hasher.md5(path) != "347c3d51b01528e5c6c071e3cff1cb55"
#             ):
#                 github.download_files_from_github(
#                     path, model_name
#                 )
#         else:
#             print("Choose between u2net, u2net_human_seg or u2netp", file=sys.stderr)

#         net.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
#         net.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
#         net.eval()
#         self.net = net

#     def forward(self, block_input: torch.Tensor):
#         image_data = block_input.permute(0, 3, 1, 2)
#         original_shape = image_data.shape[2:]
#         image_data = torch.nn.functional.interpolate(image_data, (320, 320), mode='bilinear')
#         image_data = (image_data / 255 - 0.485) / 0.229
#         out = self.net(image_data)[0][:, 0:1]
#         ma = torch.max(out)
#         mi = torch.min(out)
#         out = (out - mi) / (ma - mi) * 255
#         out = torch.nn.functional.interpolate(out, original_shape, mode='bilinear')
#         out = out[:, 0]
#         out = out.to(dtype=torch.uint8, device=torch.device('cpu'), non_blocking=True).detach()
#         return out


# def alpha_matting_cutout(
#     img,
#     mask,
#     foreground_threshold,
#     background_threshold,
#     erode_structure_size,
#     base_size,
# ):
#     size = img.size

#     img.thumbnail((base_size, base_size), Image.LANCZOS)
#     mask = mask.resize(img.size, Image.LANCZOS)

#     img = np.asarray(img)
#     mask = np.asarray(mask)

#     # guess likely foreground/background
#     is_foreground = mask > foreground_threshold
#     is_background = mask < background_threshold

#     # erode foreground/background
#     structure = None
#     if erode_structure_size > 0:
#         structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int64)

#     is_foreground = binary_erosion(is_foreground, structure=structure)
#     is_background = binary_erosion(is_background, structure=structure, border_value=1)

#     # build trimap
#     # 0   = background
#     # 128 = unknown
#     # 255 = foreground
#     trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
#     trimap[is_foreground] = 255
#     trimap[is_background] = 0

#     # build the cutout image
#     img_normalized = img / 255.0
#     trimap_normalized = trimap / 255.0

#     alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
#     foreground = estimate_foreground_ml(img_normalized, alpha)
#     cutout = stack_images(foreground, alpha)

#     cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
#     cutout = Image.fromarray(cutout)
#     cutout = cutout.resize(size, Image.LANCZOS)

#     return cutout


# def naive_cutout(img, mask):
#     empty = Image.new("RGBA", (img.size), 0)
#     cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
#     return cutout


# def get_model(model_name):
#     if model_name == "u2netp":
#         return detect.load_model(model_name="u2netp")
#     if model_name == "u2net_human_seg":
#         return detect.load_model(model_name="u2net_human_seg")
#     else:
#         return detect.load_model(model_name="u2net")


# def remove(
#     data,
#     model_name="u2net",
#     alpha_matting=False,
#     alpha_matting_foreground_threshold=240,
#     alpha_matting_background_threshold=10,
#     alpha_matting_erode_structure_size=10,
#     alpha_matting_base_size=1000,
# ):
#     model = get_model(model_name)
#     img = Image.open(io.BytesIO(data)).convert("RGB")
#     mask = detect.predict(model, np.array(img)).convert("L")

#     if alpha_matting:
#         cutout = alpha_matting_cutout(
#             img,
#             mask,
#             alpha_matting_foreground_threshold,
#             alpha_matting_background_threshold,
#             alpha_matting_erode_structure_size,
#             alpha_matting_base_size,
#         )
#     else:
#         cutout = naive_cutout(img, mask)

#     bio = io.BytesIO()
#     cutout.save(bio, "PNG")

#     return bio.getbuffer()


# def iter_frames(path):
#     return mpy.VideoFileClip(path).resize(height=320).iter_frames(dtype="uint8")


# @torch.no_grad()
# def remove_many(image_data: typing.List[np.array], net: Net):
#     image_data = np.stack(image_data)
#     image_data = torch.as_tensor(image_data, dtype=torch.float32, device=DEVICE)
#     return net(image_data).numpy()
