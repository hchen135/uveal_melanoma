import json
import os
from glob import glob
all_Slides = ["Slide 2",
"Slide 3",
"Slide 4",
"Slide 5",
"Slide 6",
"Slide 7",
"Slide 8",
"Slide 9",
"Slide 10",
"Slide 11",
"Slide 12",
"Slide 13",
"Slide 14",
"Slide 15",
"Slide 17",
"Slide 18",
"Slide 19",
"Slide 21",
"Slide 22",
"Slide 23",
"Slide 24",
"Slide 25",
"Slide 26",
"Slide 27",
"Slide 28",
"Slide 29",
"Slide 30",
"Slide 31",
"Slide 33",
"Slide 34",
"Slide 35",
"Slide 36",
"Slide 37",
"Slide 38",
"Slide 40",
"Slide 41",
"Slide 42",
"Slide 43",
"Slide 45",
"Slide 47",
"Slide 48",
"Slide 49",
"Slide 50",
"Slide 52",
"Slide 53",
"Slide 55",
"Slide 56",
"Slide 59",
"Slide 60",
"Slide 61",
"Slide 63",
"Slide 64",
"Slide 65",
"Slide 66",
"Slide 67",
"Slide 68",
"Slide 69",
"Slide 70",
"Slide 71",
"Slide 72",
"Slide 74",
"Slide 75",
"Slide 76",
"Slide 77",
"Slide 78",
"Slide 79",
"Slide 80",
"Slide 81",
"Slide 82",
"Slide 83",
"Slide 84",
"Slide 85",
"Slide 86",
"Slide 88",
"Slide 89",
"Slide 90",
"Slide 91",
"Slide 92",
"Slide 93",
"Slide 95",
"Slide 96",
"Slide 97",
"Slide 98",
"Slide 99",
"Slide 100",
]
test_no_survival = [
"Slide 29",
"Slide 77",
"Slide 82",
"Slide 61",
"Slide 100",
"Slide 72"
]

test_yes_survival = [
"Slide 4",
"Slide 7",
"Slide 22",
"Slide 38",
"Slide 47",
"Slide 49",
"Slide 56",
"Slide 63",
"Slide 64",
"Slide 67"
]

train_no_survival = [
"Slide 86",
"Slide 26",
"Slide 71",
"Slide 55",
"Slide 76",
"Slide 95",
"Slide 74",
"Slide 91",
"Slide 75",
"Slide 92",
"Slide 59",
"Slide 53",
"Slide 81",
"Slide 98"
]

train_yes_survival = [i for i in all_Slides if i not in train_no_survival and i not in test_yes_survival and i not in test_no_survival]

print(len(all_Slides))
print(len(test_yes_survival))
print(len(test_no_survival))
print(len(train_no_survival))
print(len(train_yes_survival))

DATA_DIR = "/data/datasets/UM/data/HCI_ROI_extraction"

train_dict = {}
test_dict = {}

train_no_survival_image_num = 0
train_yes_survival_image_num = 0
for i in train_no_survival:
	train_dict[i] = 0 
	train_no_survival_image_num += len(glob(os.path.join(DATA_DIR,i,"*")))
for i in train_yes_survival:
	train_dict[i] = 1 
	train_yes_survival_image_num += len(glob(os.path.join(DATA_DIR,i,"*")))
for i in test_no_survival:
	test_dict[i] = 0 
for i in test_yes_survival:
	test_dict[i] = 1 

print("train no survival tile image number:",train_no_survival_image_num)
print("train yes survival tile image number:",train_yes_survival_image_num)

with open("train_subject.json","w") as a:
	json.dump(train_dict,a,indent=4)
with open("test_subject.json","w") as a:
	json.dump(test_dict,a,indent=4)
