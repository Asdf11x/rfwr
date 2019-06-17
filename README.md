# Implementation of Constructive Incremental Learning From Only Local Information for open gym CartPole

An implementation try of the paper Constructive Incremental Learning From Only Local Information of S. Schaal and C. Atkeson.

Current State:

In the Python script linear_regression.py open gym CartPole is used as simulation. A stable baseline algorithm is used to generate a data set (Not implemented yet), which can be imitated by the proposed Receptive Field Weighted Regression algorithm (rfwr) of the paper mentioned above. 

The Matlab script rfwr.m contains the whole algorithm presented in the paper and can be learned with test_rfwr_nD.m. 

The current goal is to use the algorothm of the matlab scripts to learn the dataset and produce predictions which can be used again in the python script. 

linear_regression.py
rfwr.m
test_rfwr_nD.m