# ibug.face_pose_augmentation
A toolbox for face pose augmentation based on [3DDFA](https://ieeexplore.ieee.org/iel7/34/4359286/08122025.pdf) \[1\].

## Prerequisites
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [Sciypy](https://www.scipy.org/): `$pip3 install scipy`
* [PyTorch](https://pytorch.org/): `$pip3 install torch torchvision`
* [OpenCV](https://opencv.org/): `$pip3 install opencv-python`
* [python-igraph](https://igraph.org/python/): `$pip3 install python-igraph` __Note__: On Windows, you may need to use conda to install this package by running `conda install -c conda-forge python-igraph`
* [matplotlib](https://matplotlib.org/): `$pip3 install matplotlib`
* [shapely](https://github.com/Toblerity/Shapely): `$pip3 install shapely`
* [cython](https://cython.org/): `$pip3 install cython`
* [ibug.face_detection](https://github.com/hhj1897/face_detection) (only needed by the test script): See this repository for details: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).
* [ibug.face_alignment](https://github.com/hhj1897/face_alignment) (only needed by the test script): See this repository for details: [https://github.com/hhj1897/face_alignment](https://github.com/hhj1897/face_alignment).

## How to Install
```
git clone --recurse-submodules https://github.com/hhj1897/face_pose_augmentation.git
cd face_pose_augmentation
pip install -r requirements.txt
pip install -e .
```
__Note__: On Windows, you may need to install python-igraph (one of the dependencies) through conda by running`conda install -c conda-forge python-igraph`

## How to Use
TODO

## References
\[1\] Zhu, Xiangyu, Xiaoming Liu, Zhen Lei, and Stan Z. Li. "[Face alignment in full pose range: A 3d total solution.](https://ieeexplore.ieee.org/iel7/34/4359286/08122025.pdf)" _IEEE transactions on pattern analysis and machine intelligence_ 41, no. 1 (2017): 78-92.
