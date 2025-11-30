# Download Examples

The tests for SAM make use of the same examples for testing as the ones used in the grasp tests. To download them, follow the instructions in the corresponding [`README.md`](../grasp/examples/README.md). 


# Run Test

Firstly, make sure that you have already downloaded one of the SAM  pre-trained weights available from Meta. 
To download the pre-trained weights, follow the insturctions given in the following [`README.md`](../../obj_manipulation/sam/models/README.md) file. 
Then, you can run the following SAM test scripts to ensure that the model works as expected.

The first script tests the model in `select` mode, where the user is prompted for a set of positive and negative point labels before mask prediction.
Positive labels indicate to the model points belonging to the target object while negative ones indicate points that do not and therefore should be ignored by the model.
Typically, one or two positive and negative point labels are sufficient for getting a very accurate segmentation mask for most objects.

Meanwhile, the second script tests the model in `auto` mode, where points are sampled on a grid and the model predicts masks for each of these points.
Afterwards, these masks are filtered according to their associated confidence scores, stability under small changes to the confidence threshold and de-duplicated through NMS. 

```bash
# Setup ROS package environment variables
cd /catkin_ws
catkin_make
source devel/setup.bash

# Run test scripts
cd /catkin_ws/src/obj_manipulation
python3 tests/sam/sam_select_test.py -f 0.npy
python3 tests/sam/sam_auto_test.py -f 0.npy
```

# Expected Results

The following figures represents a sample of the expected results in `select` and `auto` modes respectively.
<p align="center">
  <img src="../..//images/SAMSelect.png" alt="SAM Select" height="200"/>
  <img src="../..//images/SAMAuto.png" alt="SAM Auto" height="200"/>
</p>