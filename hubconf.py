import torch


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


def transforms_cv2():
    from torchvision.transforms import Compose
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    import cv2

    class BGRToRGBFloat():
        def __call__(self, sample):
            sample["image"] = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB) / 255.0
            return sample

    transform = Compose([
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
    ])
    return transform


if __name__ == "__main__":
    import argparse
    import cv2
    import torch.nn.functional as F
    import numpy as np

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output image file")
    parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"],
                        help="encoder")
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    args = parser.parse_args()

    model = torch.hub.load("./", "DepthAnything", encoder=args.encoder, source="local", trust_repo=True).cuda()
    transform = torch.hub.load("./", "transforms_cv2", source="local", trust_repo=True)

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).cuda()

    if args.fp16:
        model = model.half()
        image = image.half()

    with torch.inference_mode():
        depth = model(image)
        depth = F.interpolate(depth.unsqueeze(0), (h, w), mode='bilinear', align_corners=False)
        depth = depth.squeeze(dim=[0, 1])
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    cv2.imwrite(args.output, depth_color)
