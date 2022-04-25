import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import detectron2
import os
import glob
import time
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

setup_logger()
device=torch.cuda.device("cuda")


register_coco_instances('self_coco_train', {},
                        './Data/ALL_S280_COCO/annotations.json',
                       './Data/ALL_S280_COCO')
register_coco_instances(name = "F4", metadata={},
                        json_file = "Data/Fold/F4/annotations.json", image_root = "Data/Fold/F4")

## Detection: COCO-Detection/faster_rcnn_R_101_C4_3x.yaml
## Detection: COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
## Segmentation: COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml
## Segmentation: COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("self_coco_train",)
cfg.DATASETS.TEST = ("self_coco_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
num_gpu = 1
bs = (num_gpu * 2)
cfg.SOLVER.BASE_LR = 0.0002 * bs / 16  # pick a good LR
cfg.SOLVER.MAX_ITER = 80000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [48], [96], [216], [480]]  # One size for each in feature map
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1, 2, 5, 10, 25, 50, 60, 70]]  # Three aspect ratios (same for all in feature maps)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.OUTPUT_DIR = "./output/"


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


evaluator = COCOEvaluator('self_coco_val', cfg, False, output_dir="/home/hsin/PycharmProjects/Lin/output")
val_loader = build_detection_test_loader(cfg, 'self_coco_val')
inference_on_dataset(trainer.model, val_loader, evaluator)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set the testing threshold for this model
cfg.DATASETS.TEST = ("self_coco_val", )
predictor = DefaultPredictor(cfg)


t1 = time.time()
count=0
for d in glob.glob("Dataset_COCO/JPEGImages/*.jpg"):
    count+=1
    print(d)
    im = cv2.imread(d)
    bg = np.zeros((480, 640, 3))
    outputs = predictor(im)
    print(outputs["instances"].pred_boxes)

    v = Visualizer(bg[:, :, ::-1],
                   metadata=coco_val_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.SEGMENTATION  # remove the colors of unsegmented pixels
    )
    
t2=time.time()
print("----------------------------------------------------------------------------------------------")
print("Time Usage per image: " + str((t2-t1)/count) + "Seconds")