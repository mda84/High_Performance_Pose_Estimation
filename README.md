# High Performance Pose Estimation

It's the final project of CMPT-733. The report is available [here](https://github.com/mda84/High_Performance_Pose_Estimation/blob/main/Final%20Project%20Report%20-%20High%20Performance%20Body%20Pose%20Estimation.pdf).
For deploying the models, first you need to download the .om files from [here](https://drive.google.com/drive/folders/1LpXBeGYvEUo0QCnBCH6-RUOvFELhx8c7?usp=sharing).

## OpenPose
Download the OpenPose_for_TensorFlow_BatchSize_1.om and video_pose_3d.om files and put them in this folder:
>High_Performance_Pose_Estimation/OpenPose/model/

Then being in High_Performance_Pose_Estimation/OpenPose/ folder, run the following command:
```
python run.py --input data/1.jpg
```

## Lightweight OpenPose
Download the OpenPose_light.om and video_pose_3d.om files and put them in this folder:
>High_Performance_Pose_Estimation/LightweightOpenPose/model/

Then being in High_Performance_Pose_Estimation/LightweightOpenPose/ folder, run the following command:
```
python run.py --input data/1.jpg
```

## UniPose
Download the UniPose_MPII.om and video_pose_3d.om files and put them in this folder:
>High_Performance_Pose_Estimation/UniPose/model/

Then being in High_Performance_Pose_Estimation/UniPose/ folder, run the following command:
```
python run.py --input data/1.jpg
```

## MHFormer
Download the model_4294.om file and put it in this folder:
>High_Performance_Pose_Estimation/MHFormer/checkpoint/

Also, download the pose_hrnet_w48_384x288.om and yolov3.om files and put them in this folder:
>High_Performance_Pose_Estimation/MHFormer/demo/lib/checkpoint/

Then being in High_Performance_Pose_Estimation/MHFormer/ folder, run the following command:
```
python demo/vis.py --video 1.jpg
```
