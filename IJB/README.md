# IJB Eval Tools

## Usage

##### Step1. Prepare dataset and tools.
1. Download IJB(B/C) package provided by [`insightface`](https://github.com/deepinsight/insightface/tree/master/evaluation/IJB).
 (please follow LICENSE strictly)
2. Unzip this package and get file structure as below:
```
IJB_root
├── IJBB
│   ├── loose_crop\
│   └── meta
│       ├── ijbb_1N_gallery_S1.csv
│       ├── ijbb_1N_gallery_S2.csv
│       ├── ijbb_1N_probe_mixed.csv
│       ├── ijbb_face_tid_mid.txt
│       ├── ijbb_name_5pts_score_retina.txt
│       ├── ijbb_name_5pts_score.txt
│       ├── ijbb_name_box_score_5pts.txt
│       └── ijbb_template_pair_label.txt
│   
└── IJBC
    ├── loose_crop\
    └── meta
        ├── ijbc_1N_gallery_G1.csv
        ├── ijbc_1N_gallery_G2.csv
        ├── ijbc_1N_probe_mixed.csv
        ├── ijbc_face_tid_mid.txt
        ├── ijbc_name_5pts_score_retina.txt
        ├── ijbc_name_5pts_score.txt
        ├── ijbc_name_box_score_5pts.txt
        └── ijbc_template_pair_label.txt
```

##### Step2. Running IJB Evaluation.
1. [Optional] Align image

    We could use `align_img.py` to create aligned images if you want to eval this many times.
    Otherwise each eval will align img again.
    
    Please make sure `IJB_root` have enough space to save align images.
    ```python
    usage: align_img.py
                --target             # IJBB or IJBC to process.
                --root-path          # IJB_root path in Step1.
                --retina-landmark    # Whether use retina landmark.
                --batch-size         # Batch size used in dataloader.
                --num-workers        # Num workers used in dataloader.
    ```
    
    An example:
    ```shell script
    python align_img.py --target IJBC
                        --root-path ../../IJB_release
                        --batch-size 1024
                        --num-workers 24
                        --retina-landmark
    
    ```
    
    This script will create `with_crop` or `with_crop_retina` folder under IJB_root dir.

2. For IJB 1:1 Evaluation we need two steps.

    - 2.1 First computing verification scores between template pairs.
        - We First use `IJB_11.py` generate scores and save.
            ```shell script
            usage: IJB_11.py 
                    --model-path        # TorchScript model path.
                    --batch-size        # Batch size used in dataloader.
                    --target            # IJBB or IJBC to eval.
                    --root-path         # IJB_root path in Step1.
                    --use-aligned-image # Use aligned image directly.
                    --retina-landmark   # Use retina landmark.
                    --flip              # Use flip test(F1).
                    --det-score         # Use det score(D1).
                    --num-workers       # Num workers used in dataloader.
                    --save-path         # Save result path.
            ```
            An example here.
            ```python
              python IJB_11.py  --batch-size 256 \
                                --target IJBC \
                                --root-path ../../IJB_release \
                                --use-aligned-image \
                                --retina-landmark \
                                --num-workers 6 \
                                --model-path  ../../mobilefacenet_glint360k.pt \
                                --save-path   ../../mobilefacenet_ijb
            ```
            This also create a `label.npy` in `meta` folder will be used in 2.2.
            
    - 2.2 Show tpr@fpr table and save ROC curve.
        - Use `plot_ijb11_result.py` to show and save IJB 1:1 result.
            ```shell script
              usage: plot_ijb11_result.py
                         --root-path     # IJB_root path in Step1.
                         --target        # IJBB or IJBC to eval.
                         --results-path  # 2.1 save path.
                         --save-path     # Save result path.
            ```
            And we will get result like this.
            ```shell script
            +------------------------------------------------+-------+-------+--------+-------+-------+-------+
            |                    Methods                     | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
            +------------------------------------------------+-------+-------+--------+-------+-------+-------+
            |  VGG2-ResNet50-ArcFace-TestMode(N0D1F0)-IJBC   | 76.37 | 87.33 | 92.58  | 96.17 | 98.31 | 99.36 |
            | MS1MV2-ResNet100-ArcFace-TestMode(N1D1F0)-IJBC | 89.43 | 94.13 | 96.10  | 97.35 | 98.29 | 99.05 |
            | MS1MV2-ResNet100-ArcFace-TestMode(N1D0F0)-IJBC | 89.06 | 93.94 | 96.03  | 97.31 | 98.29 | 99.04 |
            |  VGG2-ResNet50-ArcFace-TestMode(N1D0F0)-IJBC   | 68.63 | 85.54 | 91.99  | 95.86 | 98.20 | 99.34 |
            | MS1MV2-ResNet100-ArcFace-TestMode(N0D1F0)-IJBC | 87.95 | 93.87 | 95.91  | 97.31 | 98.24 | 99.04 |
            |  VGG2-ResNet50-ArcFace-TestMode(N0D1F2)-IJBC   | 76.68 | 87.96 | 92.89  | 96.36 | 98.40 | 99.41 |
            |  VGG2-ResNet50-ArcFace-TestMode(N0D0F0)-IJBC   | 70.81 | 86.12 | 92.14  | 95.95 | 98.23 | 99.34 |
            |  VGG2-ResNet50-ArcFace-TestMode(N1D1F1)-IJBC   | 75.86 | 87.17 | 92.53  | 96.20 | 98.36 | 99.37 |
            | MS1MV2-ResNet100-ArcFace-TestMode(N1D1F2)-IJBC | 89.85 | 94.47 | 96.28  | 97.53 | 98.36 | 99.08 |
            | MS1MV2-ResNet100-ArcFace-TestMode(N1D1F1)-IJBC | 89.97 | 94.34 | 96.18  | 97.44 | 98.32 | 99.07 |
            | MS1MV2-ResNet100-ArcFace-TestMode(N0D0F0)-IJBC | 86.25 | 93.15 | 95.65  | 97.20 | 98.18 | 99.01 |
            | MS1MV2-ResNet100-ArcFace-TestMode(N0D1F2)-IJBC | 88.29 | 94.00 | 96.07  | 97.46 | 98.33 | 99.10 |
            |  VGG2-ResNet50-ArcFace-TestMode(N1D1F0)-IJBC   | 74.70 | 86.75 | 92.45  | 96.10 | 98.30 | 99.35 |
            |  VGG2-ResNet50-ArcFace-TestMode(N1D1F2)-IJBC   | 74.44 | 87.51 | 92.79  | 96.35 | 98.41 | 99.39 |
            +------------------------------------------------+-------+-------+--------+-------+-------+-------+
            ```

3. For IJB 1:N Evaluation we could get result directly.
    - We use `IJB_1N.py` to get 1:N results.
        ```shell script
        usage: IJB_1N.py
                --model-path          # TorchScript model path.
                --batch-size          # Batch size used in dataloader.
                --target              # IJBB or IJBC to eval.
                --root-path           # IJB_root path in Step1.
                --use-aligned-image   # Use aligned image directly.
                --retina-landmark     # Use retina landmark.
                --flip                # Use flip test(F1).
                --det-score           # Use det score(D1).
                --num-workers         # Num workers used in dataloader.
        ```
        Just like `IJB_11.py` but won't save anything.
        
        Then we will get results like this.
        ```shell script
        Start evaluation.
        similarity shape (19593, 3531)
        top1 = 0.9527892614709335
        top5 = 0.9673352728015108
        top10 = 0.9719287500637983
        neg_sims num = 69163290
        after sorting , neg_sims num = 1960
        far = 0.0100000000 pr = 0.8925126321 th = 0.5405273438
        far = 0.1000000000 pr = 0.9386515592 th = 0.4494628906
        ```