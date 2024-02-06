import torch
from os import path


def _load_state_dict(encoder, **kwargs):
    assert encoder in {"vits", "vitb", "vitl"}
    file_name = f"depth_anything_{encoder}14.pth"
    url = f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/{file_name}?download=true"
    state_dict = torch.hub.load_state_dict_from_url(url, file_name=file_name,
                                                    weights_only=True, map_location=torch.device("cpu"))
    return state_dict


def DepthAnything(encoder, localhub=True):
    from depth_anything.dpt import DPT_DINOv2

    if encoder == 'vits':
        depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384],
                                    localhub=localhub)
    elif encoder == 'vitb':
        depth_anything = DPT_DINOv2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768],
                                    localhub=localhub)
    elif encoder == 'vitl':
        depth_anything = DPT_DINOv2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024],
                                    localhub=localhub)
    else:
        raise ValueError(f"Unknown encoder {encoder}")

    depth_anything.load_state_dict(_load_state_dict(encoder), strict=True)

    return depth_anything


def DepthAnythingMetricDepth(model_type="indoor", remove_prep=True):
    model = torch.hub.load(path.join(path.dirname(__file__), "metric_depth"),
                           "DepthAnythingMetricDepth",
                           model_type=model_type, remove_prep=remove_prep,
                           source="local")
    return model


def transforms_cv2():
    from torchvision.transforms import Compose
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    import cv2

    class PackImage():
        def __call__(self, image):
            return {"image": image}

    class UnpackImage():
        def __call__(self, sample):
            return sample["image"]

    class BGRToRGBFloat():
        def __call__(self, sample):
            sample["image"] = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB) / 255.0
            return sample

    class ToTensor():
        def __call__(self, image):
            return torch.from_numpy(image)

    transform = Compose([
        PackImage(),
        BGRToRGBFloat(),
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        UnpackImage(),
        ToTensor(),
    ])
    return transform


def transforms_pil(lower_bound=518):
    import torchvision.transforms.functional as TF
    from torchvision.transforms import Compose, Normalize, ToTensor, InterpolationMode

    class ModConstraintLowerBoundResize():
        def __init__(self, lower_bound=518, ensure_multiple_of=14):
            self.lower_bound = lower_bound
            self.ensure_multiple_of = ensure_multiple_of

        def __call__(self, x):
            if torch.is_tensor(x):
                # CHW tensor
                h, w = x.shape[-2:]
            else:
                # PIL.Image.Image
                h, w = x.height, x.width
            if w < h:
                scale_factor = self.lower_bound / w
            else:
                scale_factor = self.lower_bound / h
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            new_h -= new_h % self.ensure_multiple_of
            new_w -= new_w % self.ensure_multiple_of
            if new_h < self.lower_bound:
                new_h = self.lower_bound
            if new_w < self.lower_bound:
                new_w = self.lower_bound
            x = TF.resize(x, (new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)
            return x

    transform = Compose([
        ModConstraintLowerBoundResize(lower_bound, ensure_multiple_of=14),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def _test_run():
    import argparse
    import torch.nn.functional as F
    import numpy as np

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output image file")
    parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"],
                        help="encoder for relative depth model")
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--remote", action="store_true", help="use remote repo")
    parser.add_argument("--reload", action="store_true", help="reload remote repo")
    parser.add_argument("--pil", action="store_true", help="use PIL instead of OpenCV")
    parser.add_argument("--metric", action="store_true", help="use metric depth model")
    parser.add_argument("--metric-model-type", default="indoor", choices=["indoor", "outdoor"],
                        help="model_type for metric depth")
    args = parser.parse_args()

    if not args.remote:
        if not args.metric:
            model = torch.hub.load(".", "DepthAnything", encoder=args.encoder,
                                   source="local", trust_repo=True).cuda()
        else:
            model = torch.hub.load(".", "DepthAnythingMetricDepth", model_type=args.metric_model_type,
                                   source="local", trust_repo=True).cuda()
        if args.pil:
            transforms = torch.hub.load(".", "transforms_pil", source="local", trust_repo=True)
        else:
            transforms = torch.hub.load(".", "transforms_cv2", source="local", trust_repo=True)
    else:
        force_reload = bool(args.reload)
        if not args.metric:
            model = torch.hub.load("nagadomi/Depth-Anything_iw3", "DepthAnything",
                                   encoder=args.encoder,
                                   force_reload=force_reload, trust_repo=True).cuda()
        else:
            model = torch.hub.load("nagadomi/Depth-Anything_iw3", "DepthAnythingMetricDepth",
                                   model_type=args.metric_model_type,
                                   force_reload=force_reload, trust_repo=True).cuda()
        if args.pil:
            transforms = torch.hub.load("nagadomi/Depth-Anything_iw3", "transforms_pil", trust_repo=True)
        else:
            transforms = torch.hub.load("nagadomi/Depth-Anything_iw3", "transforms_cv2", trust_repo=True)

    if args.pil:
        import PIL
        image = PIL.Image.open(args.input).convert("RGB")
        h, w = image.height, image.width
        image = transforms(image).unsqueeze(0).cuda()
    else:
        import cv2
        image = cv2.imread(args.input, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        image = transforms(image).unsqueeze(0).cuda()

    if args.fp16:
        model = model.half()
        image = image.half()

    with torch.inference_mode():
        depth = model(image)
        if not args.metric:
            depth = depth.unsqueeze(0)
        else:
            depth = depth["metric_depth"].neg()
        depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)
        depth = depth.squeeze(dim=[0, 1])
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    if args.pil:
        import torchvision.transforms.functional as TF
        depth = TF.to_pil_image(depth.cpu())
        # No color map
        depth.save(args.output)
    else:
        depth = (depth * 255).cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(args.output, depth_color)


if __name__ == "__main__":
    _test_run()
