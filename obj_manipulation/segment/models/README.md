# Download Model

You can download the pre-trained model's weights from the following [link](https://drive.google.com/drive/folders/1_rh8EcGW7P9Vo4pw2w09rKpbWrmR3CKu?usp=sharing). 
Download the files named `DSN.pth` and `RRN.pth` and place them inside this directory.


# Load Weights

The implemented `InstanceSegmentationFull` class automatically handles loading both models' weights. 
To load the instance segmentation model's weights, only the `load()` method needs to be invoked as shown in the following example:

```python
from obj_manipulation.segment import InstanceSegmentationFull

# Load instance segmentation module's configuration
config = ...

# Initialize instance segmentation module and load its weights
ins_seg = InstanceSegmentationFull(config)
ins_seg.load()
```