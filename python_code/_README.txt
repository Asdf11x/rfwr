08.07.2019 - Peter Muschick

Readme for submission of Imitation Learning Seminar:

Included files:
HS-IL-Jupyter-Notebook-Peter_Muschick 	(Folder containing jupyter notebook)
_README.txt 							(current file)
full_dataset_120k.csv 					(Dataset containing 120k lines of training data, created by harsha_evolution.py)
harsha_evolution.py 					(Evolutionary algorithm used to create training data)
harsha_evolution_cropped.csv 			(Dataset containing 4,8k lines of training data, created by harsha_evolution.py, only first 100 timestop used)
HS-IL-Presentation-Peter_Muschick.pdf	(file containing the presentation)
linear.csv								(test file containing points in linear order)
linear_plateau.csv						(test file containing points in linear order with a short pleateau)
lwpr_algorithm.py						(lwpr algorithm file)
main.py									(The main python script containing the cartpole open gym environment)
networking.py							(contains the UDP networking part for main.py)
sharvar_keras.py						(generates training data with an Proximal Policy Optimization algorithm)
sharvar_keras_data.csv					(created data from sharvar_keras.py)
sinus_noise.csv							(creates scattered sinus points)

Remarks/explanation to specific files:

main.py.
	- This file needs to be executed with python 3.x (open gym is only running with python 3.x)

lwpr_algorithm.py
	- https://github.com/jdlangs/lwpr
	- Check their README.txt
	- Basically a python 2.7 interpreter (32 bit) with a few packages (described in their README.txt) and GCC installed on top of it needs to run it