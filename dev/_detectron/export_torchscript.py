import argparse
import os
import torch

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

class ScriptableDetectron2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, image_bgr_u8: torch.Tensor):
        if image_bgr_u8.dtype != torch.uint8:
            image_bgr_u8 = image_bgr_u8.to(torch.uint8)

        img = image_bgr_u8.permute(2, 0, 1).to(dtype=torch.float32)
        inputs = [{"image": img, "height": image_bgr_u8.shape[0], "width": image_bgr_u8.shape[1]}]

        outputs = self.model(inputs)
        inst = outputs[0]["instances"]

        boxes = inst.pred_boxes.tensor
        scores = inst.scores
        classes = inst.pred_classes.to(torch.int64)

        out = {
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
        }

        if inst.has("pred_masks"):
            m = inst.pred_masks
            out["masks"] = m

        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to detectron2 config.yaml")
    ap.add_argument("--weights", required=True, help="Path to model_final.pth")
    ap.add_argument("--out", required=True, help="Output TorchScript file (.pt)")
    ap.add_argument("--device", default="cpu", help="cpu recommended for export")
    ap.add_argument("--max-dets", type=int, default=400)
    ap.add_argument("--score-thresh", type=float, default=0.0)
    ap.add_argument("--example-h", type=int, default=1024)
    ap.add_argument("--example-w", type=int, default=1024)
    args = ap.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(args.config)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(args.weights)

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.TEST.DETECTIONS_PER_IMAGE = args.max_dets
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh

    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    model.eval()
    model.to(args.device)

    wrapped = ScriptableDetectron2(model).eval().to(args.device)

    example = torch.zeros((args.example_h, args.example_w, 3), dtype=torch.uint8, device=args.device)

    ts = torch.jit.trace(wrapped, (example,), strict=False)
    ts.save(args.out)
    print(f"Saved TorchScript: {args.out}")

if __name__ == "__main__":
    main()