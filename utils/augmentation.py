import io
import torch
from PIL import Image
from torchvision.transforms import functional as F

class RandomJPEGReencode:
    def __init__(self, qmin=40, qmax=80, p=0.5):
        self.qmin = qmin
        self.qmax = qmax
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img
        q = int(torch.randint(self.qmin, self.qmax + 1, (1,)).item())
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        img_jpeg = Image.open(buf).convert("RGB")
        return img_jpeg

class RandomCenterCropResize:
    def __init__(self, scale_min=0.85, scale_max=1.0, out_size=(299,299)):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.out_size = out_size

    def __call__(self, img):
        w, h = img.size
        scale = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()

        new_w = int(w * scale)
        new_h = int(h * scale)

        left   = (w - new_w) // 2
        top    = (h - new_h) // 2
        right  = left + new_w
        bottom = top + new_h

        img_cropped = img.crop((left, top, right, bottom))
        img_resized = F.resize(img_cropped, self.out_size)
        return img_resized
