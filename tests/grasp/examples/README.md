# Download Examples

You can download the test examples for the grasp estimation module from the following [link](https://drive.google.com/drive/folders/1AtZvhrpF-UjZ15r1o7QjC5r1mnE5miKG?usp=sharing).
Download all files and place them inside this `examples` directory.


# Run Test

Firstly, make sure that you have already downloaded the grasp estimation module's pre-trained weights or have trained it yourself and saved its weights. 
To download the pre-trained weights, follow the insturctions given in the following [`README.md`](../../../obj_manipulation/grasp/models/README.md) file.

Additionally, since grasp estimation relies on instance segmentation, you must first ensure that the instance segmentation module works as expected.
To do this, follow the instructions given in the following [`README.md`](../../segment/examples/README.md) file.


Then, you can run the following grasp estimation test scripts to ensure that the model works as expected.
The first script uses the point cloud filter based on the the `UOIS` instance segmentation module, while the second relies on the `SAM` module for instance segmentation.

```bash
# Setup ROS package environment variables
cd /catkin_ws
catkin_make
source devel/setup.bash

# Run test scripts
cd /catkin_ws/src/obj_manipulation
python3 tests/grasp/grasp_uois_test.py -f 0.npy
python3 tests/grasp/grasp_sam_test.py -f 0.npy
```

# Expected Results

The following figure showcases the expected grasp estimation results from the `grasp_uois_test.py` test script.
The colors of the shown grasps represent their predicted success probabilities.
All grasps are shown with a fixed grasp width set to the gripper's width.
<p align="center">
  <img src="../../../images/GraspEstimation.png" alt="Grasp Estimation" height="400"/>
</p>