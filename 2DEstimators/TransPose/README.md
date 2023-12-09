# TransPose
Here we put the code we used for estimation of 2D sequences of Human3.6M using TransPose.
## usage
Run the following command:
```
python transpose.py \
       --video-path <PATH-TO-INPUT-VIDEOS> \
       --out-pose-root <PATH-TO-OUTPUT-VIDEOS>
```
where the `--video-path` receives path to a directory containing the mp4 files and `--out-pose-root` receives path to a directory to store all the estimated pose sequences.