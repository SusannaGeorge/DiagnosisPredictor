patients = open('patients.txt','r').read()
max_length = 0
j=0
patientlist = patients.split('\n')
print(patientlist)
for i in patientlist:
    print(i)
    patient[j] = i.split()
    if len(patient[j]) > max_length:
        max_length = len(patient[j]
    j=j+1
print("count ",j)
print("max_length ", max_length)
