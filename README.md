# BrainReg

This is the code for paper Automatic Landmark Detection and Registration of Brain Cortical Surfaces via Quasi-Conformal Geometry and Convolutional Neural Network

To make use of our model, DBSnet should be run first:

```
cd DBSnet
python train.py
```

CPnet should be trained after DBSnet is trained. To load the args, code could be run as:

```
cd CPnet
python train.py -f CP_epochxxx.py
```

LDnet should be trained separately:
```
cd LDnet
python train.py
```
