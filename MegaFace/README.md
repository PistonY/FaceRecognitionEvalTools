# MegaFace Eval Tools

## Usage

##### Step1. Prepare dataset and tools.
1. Download megaface package provided by [`insightface`](https://github.com/deepinsight/insightface/tree/master/evaluation/Megaface).
 (please follow LICENSE strictly)
2. Unzip this package and get file structure as below:
    ```
    megaface_root
    ├── devkit\
    ├── facescrub_images\
    ├── facescrub_lst
    ├── facescrub_lst_all
    ├── facescrub_noises_empty.txt
    ├── facescrub_noises.txt
    ├── megaface_images\
    ├── megaface_lst
    ├── megaface_noises_empty.txt
    └── megaface_noises.txt
    ```
3. Download [opencv2.4](https://pan.baidu.com/s/1By4yIds0hEnw6_Ihh75R5w) which devkit need.
Thanks [deepinx/megaface-evaluation](https://github.com/deepinx/megaface-evaluation) provide this.
Unzip to `opencv_path/opencv2.4`.

##### Step2. Generate features.
Use `megaface.py` to generate features, params describe as bellow.
```python
usage: megaface.py 
     --megaface-root     # megaface_root in step1.
     --batch-size        # batch size to use when inference.
     --num-workers       # num_works for dataloader. 
     --output-root       # path to save feature generated.
     --model-path        # model param path.
     --remove-noise      # whether remove noise by insightface provided information.
     --flip              # whether use flip added test.
```
An example:
```shell script
python megaface.py --megaface-root ../../megaface
                    --batch-size 1024
                    --num-workers 24  
                    --output-root ../../megaface_mobilenet
                    --flip
                    --remove-noise  
                    --model-path ./params/MobileNet_Glint360K_55.pt
```
After doing this the output folder should have follow structure:
```
megaface_out
├── facescrub\
└── megaface\
```

Tips: The noise removed way used in this tools is `different` from `insightface`,
 so the final accuracy may not totally same.
##### Step3. Running MegaFace Evaluation.

Before running script provided by `devkit` and get final result, we need a `Python2` environment.

If you use `conda`, you could run `conda create -n py2 python=2.7` to create one.

Go into `devkit/experiments` and make sure you run follow script in `py2 env`.
```shell script
export LD_LIBRARY_PATH="opencv_path/opencv2.4:$LD_LIBRARY_PATH"
python -u run_experiment.py megaface_out/megaface \
                            megaface_out/facescrub \
                            _model_name \
                            megaface_out
```

Tips: You'd better use a server to run this. 
Running in personal PC may cause a disaster.

After finish running you should get similar result like this.
```
Probe score matrix size: 3530 3530
distractor score matrix size: 3530 1000000
Done loading. Time to compute some stats!
Finding top distractors!
Done sorting distractor scores
Making gallery!
Done Making Gallery!
Allocating ranks (1000080)
Rank 1: 0.952087

```

##### Plot Result
Refers [here](https://github.com/deepinx/megaface-evaluation), please feel free to explore and use it.