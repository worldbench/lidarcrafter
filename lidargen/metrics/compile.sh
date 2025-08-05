#!/bin/sh

cd modules/chamfer2D
python setup.py build_ext --inplace

cd ../chamfer3D
python setup.py build_ext --inplace

cd ../emd
python setup.py build_ext --inplace

cd ..
