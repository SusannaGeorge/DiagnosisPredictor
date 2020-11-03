import pandas as pd, ast, re

#file = open('icd_codes.txt',"r")
#contents = file.read()
#dictionary = ast.literal_eval(contents)
#file.close()

f0 = open('mixed.txt',"r")
mixed = f0.read()
f0.close()
tokensf0 = re.findall("[A-Za-z0-9]+", mixed)
print(tokensf0)

f1 = open('patients.txt',"r")
patients = f1.read()
f1.close()
tokensf1 = re.findall("[A-Za-z0-9]+", patients)
print(tokensf1)

f2 = open('visits.txt',"r")
visits = f2.read()
f2.close()
tokensf2 = re.findall("[A-Za-z0-9]+", visits)
print(tokensf2)
