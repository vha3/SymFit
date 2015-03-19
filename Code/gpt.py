from numpy import *
from scipy import *
from operator import itemgetter
from random import sample
from ast import *
import matplotlib.pyplot as plt
from treeclass_troubleshoot import *

## Start by importing the data. 
###########################################
set1 = list(loadtxt('SR_circle.txt'))
set2 = list(loadtxt('SR_div.txt'))
set3 = list(loadtxt('SR_line.txt'))
set4 = list(loadtxt('SR_div_noise.txt'))
set5 = list(loadtxt('SR_line_noise.txt'))
frankie = list(loadtxt('Frankie.txt'))
atmos = list(loadtxt('Atmosphere.txt'))

set1_dict={}
set2_dict={}
set3_dict={}
set4_dict={}
set5_dict={}
frankie_dict={}
atmos_dict={}

for i in range(len(set1)):
	set1[i]=list(set1[i])
	set2[i]=list(set2[i])
	set3[i]=list(set3[i])
	set4[i]=list(set4[i])
	set5[i]=list(set5[i])
	key=str(i)
	set1_dict[i]=set1[i]
	set2_dict[i]=set2[i]
	set3_dict[i]=set3[i]
	set4_dict[i]=set4[i]
	set5_dict[i]=set5[i]



for i in range(len(frankie)):
	frankie[i]=list(frankie[i])
	frankie_dict[i]=frankie[i]

for i in range(len(atmos)):
	atmos[i]=list(atmos[i])
	atmos_dict[i]=atmos[i]


set1=sorted(set1, reverse=False, key=itemgetter(0))
set2=sorted(set2, reverse=False, key=itemgetter(0))
set3=sorted(set3, reverse=False, key=itemgetter(0))
set4=sorted(set4, reverse=False, key=itemgetter(0))
set5=sorted(set5, reverse=False, key=itemgetter(0))
frankie=sorted(frankie, reverse=False, key=itemgetter(0))
atmos=sorted(atmos, reverse=False, key=itemgetter(0))


def initialize_population(population_size,max_constant):
	town = population()
	for i in range(population_size+1):
		citizen = person()
		citizen.root = operator_dict[random.randint(-1,8)]
		decider = random.random()
		if decider<0.5:
			citizen.left = person(random.uniform(-max_constant,max_constant))
			citizen.right = person(operator_dict[8])
		if decider>0.5:
			citizen.right = person(random.uniform(-max_constant,max_constant))
			citizen.left = person(operator_dict[8])
		town.add_person(citizen)
	town.people=town.people[1:]
	return town

#	



