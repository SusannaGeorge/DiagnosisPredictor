with open('codeicd9inputs.txt') as file_:
  test1 = file_.readlines()
ip=[[]]
#op=[[]]
for t1 in test1:
	t2=t1.split(', ')
	x=''
	for i in range(0,len(t2)):
		t3=' '.join(t2[i].split(','))
		t3=t3.replace('[',' ').replace(']','')
		x+=t3
		j=i+1
		if j<len(t2):
			t4=' '.join(t2[j].split(','))
			t4=t4.replace('[',' ').replace(']','')
			t5=t4.split()
			for k in range(0,len(t5)):
				y=x
				y+=' '+ t5[k]
				ip.append(y)

plist = open('icdbiosent2vec.txt','r').read().split('\n')
plist.remove('')
icds = [[p[0:len(p)] for p in patient.split(', ')] for patient in plist]
icdlist=[icd[0] for icd in icds]
print('length of icdlist',len(icdlist))
print('length of ip',len(ip))

i=0
j=0
with open('inputicdsentences.txt','w') as f1:
	for ipsentence in ip:
		for ipword in str(ipsentence).split():
			if ipword in icdlist:
				i+=1
				f1.write(str(ipword)+ ' ')
		f1.write('\n')
		j+=1
		print(j)
print('i',i)

with open('inputicdsentences.txt') as f2:
	test2 = list(f2)
maxim=0
for t1 in test2:
	t2=t1.split()
	if len(t2[0:-1])>maxim:
		maxim=len(t2[0:-1])
print('maxim ',maxim)		
		
		
		# t3=t2[i]
		# y=''
		# for j in range(0,len(t2[i])):
			# t3=
			# y.append()
			
	# for t in t2[1:-1]:
		# x+=' '.join(t.split(','))
		# x=x.replace('[',' ').replace(']','')
	# ip.append(x.split())
	# xy=''
	# for t in t2[-1]:
		# xy+=' '.join(t.split(','))
		# xy=xy.replace('[',' ').replace(']','')
	# op.append(xy.split())
