from enum import Enum

class ALG(Enum):
	CNN_trans = 0
	HIPT_trans = 1
	ABMIL = 2
	ADMIL = 3
	transMIL = 4
	DSMIL = 5
	CLAM_SB = 6
	CLAM_MB = 7
	CLAM = 8

class SELF_SUPERVISION(Enum):
	simCLR = 0
	MoCO = 1
	DINO = 2

class PHASE(Enum):
	train = 0
	valid = 1
	test = 2

class TISSUE_TYPE(Enum):
	Colon = 0
	Lung = 1
	Celiac = 2

if __name__ == "__main__":
	pass