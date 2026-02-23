import argparse
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine.train_loop import SimpleTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.logger import setup_logger


def split_coco_json(
    src_json: Path,
    out_train_json: Path,
    out_val_json: Path,
    val_ratio: float,
    seed: int,
    dataset_fraction: float = 1.0,
) -> Tuple[int, int]:
    with src_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = list(coco.get("images", []))
    anns = list(coco.get("annotations", []))

    if not images:
        raise ValueError(f"No images found in COCO json: {src_json}")

    rnd = random.Random(seed)

    if dataset_fraction < 1.0:
        k = max(1, int(round(len(images) * dataset_fraction)))
        images = rnd.sample(images, k)

    rnd.shuffle(images)
    n_val = max(1, int(round(len(images) * val_ratio)))
    val_images = images[:n_val]
    train_images = images[n_val:] if len(images) > n_val else images[:]

    train_ids = set(img["id"] for img in train_images)
    val_ids = set(img["id"] for img in val_images)

    def filter_anns(image_id_set):
        return [a for a in anns if a.get("image_id") in image_id_set]

    train_anns = filter_anns(train_ids)
    val_anns = filter_anns(val_ids)

    categories = coco.get("categories", [])
    info = coco.get("info", {})
    licenses = coco.get("licenses", [])

    train_coco = {
        "info": info,
        "licenses": licenses,
        "images": train_images,
        "annotations": train_anns,
        "categories": categories,
    }
    val_coco = {
        "info": info,
        "licenses": licenses,
        "images": val_images,
        "annotations": val_anns,
        "categories": categories,
    }

    out_train_json.parent.mkdir(parents=True, exist_ok=True)
    with out_train_json.open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False)
    with out_val_json.open("w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False)

    return len(train_images), len(val_images)


def _read_category_names(coco_json: Path):
    try:
        data = json.loads(coco_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    cats = data.get("categories", [])
    out = []
    try:
        cats_sorted = sorted([c for c in cats if isinstance(c, dict) and "id" in c and "name" in c], key=lambda x: int(x["id"]))
        out = [str(c["name"]) for c in cats_sorted]
    except Exception:
        out = []
    return out


class AccumSimpleTrainer(SimpleTrainer):
    """
    One Detectron2 iteration == one optimizer update.
    Internally, consumes accum_steps micro-batches (each of size cfg.SOLVER.IMS_PER_BATCH).
    This keeps Detectron2 hooks (LR scheduler, eval, checkpoints) correct.
    """

    def __init__(self, model, data_loader, optimizer, accum_steps: int):
        super().__init__(model, data_loader, optimizer)
        if accum_steps < 1:
            raise ValueError("accum_steps must be >= 1")
        self.accum_steps = accum_steps

    def run_step(self):
        assert self.model.training, "[AccumSimpleTrainer] model was changed to eval mode!"

        start = time.perf_counter()
        data_time_total = 0.0

        self.optimizer.zero_grad(set_to_none=True)

        loss_sums: Dict[str, torch.Tensor] = {}

        for _ in range(self.accum_steps):
            t0 = time.perf_counter()
            data = next(self._data_loader_iter)
            data_time_total += time.perf_counter() - t0

            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                loss_dict = {"total_loss": loss_dict}
            losses = sum(loss_dict.values())

            (losses / self.accum_steps).backward()

            for k, v in loss_dict.items():
                v_det = v.detach()
                loss_sums[k] = loss_sums.get(k, 0.0) + v_det

        self.optimizer.step()

        loss_avg = {k: (v / self.accum_steps) for k, v in loss_sums.items()}
        self._write_metrics(loss_avg, data_time_total)


class AccumAMPTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, accum_steps: int):
        super().__init__(model, data_loader, optimizer)
        if accum_steps < 1:
            raise ValueError("accum_steps must be >= 1")
        self.accum_steps = accum_steps
        self.grad_scaler = torch.amp.GradScaler(device="cuda")

    def run_step(self):
        assert self.model.training, "[AccumAMPTrainer] model was changed to eval mode!"

        data_time_total = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        loss_sums: Dict[str, torch.Tensor] = {}

        for _ in range(self.accum_steps):
            t0 = time.perf_counter()
            data = next(self._data_loader_iter)
            data_time_total += time.perf_counter() - t0

            with torch.amp.autocast(device_type="cuda"):
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    loss_dict = {"total_loss": loss_dict}
                losses = sum(loss_dict.values())

            scaled = losses / self.accum_steps
            self.grad_scaler.scale(scaled).backward()

            for k, v in loss_dict.items():
                v_det = v.detach()
                loss_sums[k] = loss_sums.get(k, 0.0) + v_det

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        loss_avg = {k: (v / self.accum_steps) for k, v in loss_sums.items()}
        self._write_metrics(loss_avg, data_time_total)


class MyTrainer(DefaultTrainer):
    def __init__(self, cfg, accum_steps: int):
        super(DefaultTrainer, self).__init__()
        setup_logger()

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if cfg.SOLVER.AMP.ENABLED:
            self._trainer = AccumAMPTrainer(model, data_loader, optimizer, accum_steps=accum_steps)
        else:
            self._trainer = AccumSimpleTrainer(model, data_loader, optimizer, accum_steps=accum_steps)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=self)
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=cfg.OUTPUT_DIR)


def _maybe_suppress_third_party_warnings(enable: bool):
    if not enable:
        return
    warnings.filterwarnings("ignore", message="torch\\.meshgrid: in an upcoming release.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="is_fx_tracing will return true.*", category=UserWarning)


def _scale_solver_schedule(cfg, new_max_iter: int):
    old_max_iter = int(cfg.SOLVER.MAX_ITER)
    if old_max_iter <= 0:
        cfg.SOLVER.MAX_ITER = new_max_iter
        return

    old_steps = list(cfg.SOLVER.STEPS) if cfg.SOLVER.STEPS is not None else []
    if old_steps:
        scaled = []
        for s in old_steps:
            ns = int(round(s * (new_max_iter / old_max_iter)))
            if 0 < ns < new_max_iter:
                scaled.append(ns)
        scaled = sorted(set(scaled))
        cfg.SOLVER.STEPS = tuple(scaled)
    if hasattr(cfg.SOLVER, "WARMUP_ITERS"):
        wi = int(cfg.SOLVER.WARMUP_ITERS)
        new_wi = int(round(wi * (new_max_iter / old_max_iter)))
        cfg.SOLVER.WARMUP_ITERS = max(0, min(new_wi, new_max_iter))


def build_cfg(args, train_name: str, val_name: str, iters_per_epoch: int, classes: List[str]):
    cfg = get_cfg()
    base_cfg = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(base_cfg))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_cfg)

    cfg.OUTPUT_DIR = args.output_dir
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)

    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

    cfg.INPUT.MIN_SIZE_TRAIN = (args.min_size,)
    cfg.INPUT.MIN_SIZE_TEST = args.min_size
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size
    cfg.INPUT.MAX_SIZE_TEST = args.max_size

    cfg.INPUT.RANDOM_FLIP = "none"

    if args.small_anchors:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4], [8], [16], [32], [64]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] * len(cfg.MODEL.ANCHOR_GENERATOR.SIZES)

    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = float(args.base_lr)

    cfg.SOLVER.AMP.ENABLED = bool(args.amp)

    if hasattr(cfg.SOLVER, "CLIP_GRADIENTS") and getattr(args, "clip_grad", 0) and args.clip_grad > 0:
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = float(args.clip_grad)
        cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    new_max_iter = args.epochs * iters_per_epoch

    _scale_solver_schedule(cfg, new_max_iter)
    cfg.SOLVER.MAX_ITER = int(new_max_iter)

    cfg.MODEL.RPN.NMS_THRESH = 0.80

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN  = 12000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST   = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 2000

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.60

    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

    cfg.TEST.DETECTIONS_PER_IMAGE = 400

    if args.ckpt_every_epoch:
        cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch
    else:
        cfg.SOLVER.CHECKPOINT_PERIOD = args.ckpt_period

    if args.eval_every_epoch:
        cfg.TEST.EVAL_PERIOD = iters_per_epoch
    else:
        cfg.TEST.EVAL_PERIOD = args.eval_period

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True, help="Root folder with images/ and annotations.json")
    parser.add_argument("--output-dir", type=str, default="output_detectron2")
    parser.add_argument("--val-ratio", type=float, default=0.01) # 0.1
    parser.add_argument("--dataset-fraction", type=float, default=1.0, help="Use only this fraction of images (0-1]")
    parser.add_argument("--seed", type=int, default=58)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ims-per-batch", type=int, default=1, help="Micro-batch size on GPU")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation micro-steps per optimizer update")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--small-anchors", action="store_true")

    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)

    parser.add_argument("--base-lr", type=float, default=2.5e-4)

    parser.add_argument("--clip-grad", type=float, default=1.0, help="Full-model gradient clipping value; set <=0 to disable")

    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--eval-period", type=int, default=0, help="0 disables periodic eval (unless --eval-every-epoch)")
    parser.add_argument("--ckpt-period", type=int, default=5000)

    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--ckpt-every-epoch", action="store_true")

    parser.add_argument("--suppress-third-party-warnings", action="store_true",
                        help="Hide noisy warnings from torch/detectron2 internals")

    args = parser.parse_args()

    _maybe_suppress_third_party_warnings(args.suppress_third_party_warnings)

    dataset_root = Path(args.dataset_root)
    src_json = dataset_root / "annotations.json"
    if not src_json.exists():
        raise FileNotFoundError(f"Missing {src_json}. Expect COCO json named annotations.json")

    cache_dir = Path(args.output_dir) / "cache_coco"
    train_json = cache_dir / "train.json"
    val_json = cache_dir / "val.json"

    n_train, n_val = split_coco_json(
        src_json=src_json,
        out_train_json=train_json,
        out_val_json=val_json,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dataset_fraction=args.dataset_fraction,
    )

    train_name = f"phylo_train_seed{args.seed}_frac{args.dataset_fraction}_val{args.val_ratio}"
    val_name = f"phylo_val_seed{args.seed}_frac{args.dataset_fraction}_val{args.val_ratio}"

    images_dir = dataset_root / "images"
    register_coco_instances(train_name, {}, str(train_json), str(images_dir))
    register_coco_instances(val_name, {}, str(val_json), str(images_dir))

    classes = _read_category_names(train_json) or ["leaf", "internal_node", "corner"]
    MetadataCatalog.get(train_name).set(thing_classes=classes)
    MetadataCatalog.get(val_name).set(thing_classes=classes)

    effective_batch = args.ims_per_batch * args.accum_steps
    iters_per_epoch = int(math.ceil(n_train / max(1, effective_batch)))

    cfg = build_cfg(args, train_name, val_name, iters_per_epoch, classes)

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(cfg.OUTPUT_DIR) / "config.yaml").write_text(cfg.dump())
    (Path(cfg.OUTPUT_DIR) / "classes.json").write_text(json.dumps(classes, indent=2))

    setup_logger(output=cfg.OUTPUT_DIR)

    trainer = MyTrainer(cfg, accum_steps=args.accum_steps)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
