# Download Model

You can download the pre-trained model's weights from the following [link](https://drive.google.com/drive/folders/1Abd8qyra9jJt35ex5rjQAGsg3ZVPG67b?usp=sharing). 
Download the file named `CGN.pt` and place it inside this directory.


# Load Weights

The implemented `GraspEstimatorCGN` class automatically handles loading model weights. 
To load the grasp estimator model's weights, only the `load()` method needs to be invoked as shown in the following example:

```python
from obj_manipulation.grasp import GraspEstimatorCGN

# Load grasp estimator's configuration
config = ...

# Initialize grasp estimator and load its pre-trained weights 
grasp_est = GraspEstimatorCGN(config)
grasp_est.load()
```