from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np
import matplotlib.pyplot as plt


def showMask(outputs):
        mask = outputs["instances"].to("cpu").get("pred_masks").numpy()
        #这里的mask的通道数与检测到的示例的个数一致，把所有通道的mask合为一个通道
        img = np.zeros((mask.shape[1],mask.shape[2]))
        for i in range(mask.shape[0]):
                img += mask[i]

        np.where(img>0,255,0)
        for line in img:
            print ('  '.join(map(str, line)))
        cv2.namedWindow("mask",0)
        cv2.imwrite("mask.jpg",img*255)
        cv2.imshow("mask",img)
        cv2.waitKey(0)


class Detector:
        def __init__(self, model_type):
                self.cfg = get_cfg()
                self.model_type = model_type

                ## Load Pretrained Model
                if model_type == "Object_Detection":     
                        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
                        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
                
                elif model_type == "Instance_Segmentation":
                        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
                        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
                
                elif model_type == "Keypoint_Detection":
                        # segment skeleton
                        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
                        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

                elif model_type == "LVIS":
                        # only segment object
                        self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
                        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
                
                elif model_type == "LVIS":
                        # only segment object
                        self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
                        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

                elif model_type == "Panoptic_Segmentation":
                        # only segment object
                        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
                        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
                self.cfg.MODEL.DEVICE = "cuda"

                self.predictor = DefaultPredictor(self.cfg)
        
        def SaveImage(self, imagePath):

                image = cv2.imread(imagePath)
                
                if self.model_type != "Panoptic_Segmentation":
                        predictions = self.predictor(image)
                        viz = Visualizer(image[:, :, ::-1],
                        metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                        scale=0.8,
                        instance_mode=ColorMode.IMAGE_BW)
                        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
                
                else:
                        predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                        viz = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                        output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
                
                cv2.imwrite("./Sample/images/Result.jpg", output.get_image()[:, :, ::-1])
                cv2.waitKey(0)
                
        def train():
                pass
        



        def SaveMask(self, imagePath):
                
                image = cv2.imread(imagePath)
                outputs = self.predictor(image) 
                mask = outputs["instances"].to("cpu").get("pred_masks").numpy()
                binary_mask = np.zeros((mask.shape[1],mask.shape[2]))
                for i in range(mask.shape[0]):
                        binary_mask += mask[i]

                np.where(binary_mask > 0, 255, 0)
                filename = './Sample/images/a.png'
                cv2.imwrite(filename, binary_mask*255)

        def onVideo(self, videoPath):
                pass


detector = Detector(model_type = "Instance_Segmentation")

detector.SaveImage("Sample/images/Input.jpg")