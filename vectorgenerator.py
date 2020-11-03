import numpy as np, pandas as pd
import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance

# stop_words = set(stopwords.words('english'))
# def preprocess_sentence(text):
    # text = text.replace('/', ' / ')
    # text = text.replace('.-', ' .- ')
    # text = text.replace('.', ' . ')
    # text = text.replace('\'', ' \' ')
    # text = text.lower()

    # tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

    # return ' '.join(tokens)
    
# plist = open('patientsicd9vectors.txt','r').read().split('\n')
# plist.remove('')
# icd_list = [patient.split()[0] for patient in plist]
# #print(icd_list)

# vocab_size=6072
# embedding_size=700

# icd_desc = pd.read_excel('CMS32_DESC_LONG_SHORT_DX.xlsx')
# icd_desc_list ={}
# test1 = icd_desc[icd_desc.DIAGNOSIS_CODE.isin(icd_list)]
# for icd in icd_list:
	# if icd in test1['DIAGNOSIS_CODE'].tolist():
		# a = np.array_str(test1.loc[test1['DIAGNOSIS_CODE']==icd,['LONG_DESCRIPTION']].values)
		# icd_desc_list[icd] = a[2:-2].encode('utf-8')

# model_path = 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
# model = sent2vec.Sent2vecModel()
# try:
    # model.load_model(model_path)
# except Exception as e:
    # print(e)
# print('model successfully loaded')

# icdvectors = []
# for key, value in icd_desc_list.items():
	# s = preprocess_sentence(str(value,'utf-8'))
	# sv = model.embed_sentence(s)
	# sentencevector  = [v1 for sv1 in sv.tolist() for v1 in sv1]
	# #sentencevector.insert(0,key)
#	icdvectors.append(sentencevector)

# with open('icdbiosent2vec.txt','w') as f1:
	# for v1 in icdvectors:
		# f1.write(str(v1))
		# f1.write('\n')

# sentence = preprocess_sentence('Breast cancers with HER2 amplification have a higher risk of CNS metastasis and poorer prognosis.')
# print(sentence)
# sentence_vector = model.embed_sentence(sentence)
# print(sentence_vector)
# print(sentence_vector.shape)

# sentence_vector1 = model.embed_sentence(preprocess_sentence('Breast cancers with HER2 amplification have a higher risk of CNS metastasis and poorer prognosis.'))
# sentence_vector2 = model.embed_sentence(preprocess_sentence('Breast cancers with HER2 amplification are more aggressive, have a higher risk of CNS metastasis, and poorer prognosis.'))

# cosine_sim = 1 - distance.cosine(sentence_vector1, sentence_vector2)
# print('cosine similarity:', cosine_sim)
plist = open('icdbiosent2vec.txt','r').read().split('\n')
plist.remove('')
icds = [[p[0:len(p)] for p in patient.split(', ')] for patient in plist]
icdlist=[icd[0] for icd in icds]
print('icd list \n', icdlist)
# pretrained_weights = [line[1:len(line)] for line in icds]
# cosinedistlist=[[]]
# with open('icdbiosent2vec_cosinedist.txt','w') as f1:
	# for i in range(0, len(icdlist)):
		# for j in range(0, len(icdlist)):
			# dist = 1 - distance.cosine(icdvectors[i],icdvectors[j])
			# f1.write(str(dist)+ ' ')
		# f1.write('\n')
