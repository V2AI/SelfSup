# faster_rcnn.res50.fpn.coco.multiscale.1x.syncbn  

seed: 50837708

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.598
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.238
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.353
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.650
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.550 | 59.809 | 41.855 | 23.774 | 41.945 | 48.761 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 53.008 | bicycle      | 29.739 | car            | 42.657 |  
| motorcycle    | 40.859 | airplane     | 58.369 | bus            | 60.521 |  
| train         | 58.182 | truck        | 33.538 | boat           | 25.290 |  
| traffic light | 27.818 | fire hydrant | 64.357 | stop sign      | 63.783 |  
| parking meter | 46.235 | bench        | 21.461 | bird           | 34.857 |  
| cat           | 59.967 | dog          | 55.376 | horse          | 52.850 |  
| sheep         | 49.031 | cow          | 53.114 | elephant       | 57.785 |  
| bear          | 62.888 | zebra        | 63.047 | giraffe        | 63.762 |  
| backpack      | 12.711 | umbrella     | 36.155 | handbag        | 13.284 |  
| tie           | 30.258 | suitcase     | 35.907 | frisbee        | 63.544 |  
| skis          | 21.467 | snowboard    | 33.252 | sports ball    | 48.123 |  
| kite          | 41.124 | baseball bat | 27.473 | baseball glove | 35.415 |  
| skateboard    | 49.033 | surfboard    | 33.083 | tennis racket  | 43.608 |  
| bottle        | 39.109 | wine glass   | 34.184 | cup            | 41.301 |  
| fork          | 29.817 | knife        | 14.860 | spoon          | 13.945 |  
| bowl          | 40.682 | banana       | 23.017 | apple          | 18.021 |  
| sandwich      | 30.446 | orange       | 29.719 | broccoli       | 19.988 |  
| carrot        | 18.481 | hot dog      | 29.861 | pizza          | 48.975 |  
| donut         | 40.916 | cake         | 33.945 | chair          | 24.699 |  
| couch         | 39.177 | potted plant | 22.986 | bed            | 39.977 |  
| dining table  | 25.003 | toilet       | 55.545 | tv             | 51.425 |  
| laptop        | 56.210 | mouse        | 58.329 | remote         | 27.846 |  
| keyboard      | 47.537 | cell phone   | 33.824 | microwave      | 52.382 |  
| oven          | 28.069 | toaster      | 44.780 | sink           | 33.973 |  
| refrigerator  | 50.924 | book         | 14.558 | clock          | 49.649 |  
| vase          | 36.613 | scissors     | 18.512 | teddy bear     | 40.968 |  
| hair drier    | 0.000  | toothbrush   | 20.829 |                |        |
