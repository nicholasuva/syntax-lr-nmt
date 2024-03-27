#! /usr/bin/env python



from tensorflow.compat.v1.keras.preprocessing.text import Tokenizer
#import baseline
#from baseline import file_to_sents_list

l1_file = 'MultiCCAligned.es-ro.es'
l2_file = 'MultiCCAligned.es-ro.ro'

l1_out = 'MCCA_es-ro_es_short_sents.txt'
l2_out = 'MCCA_es-ro_ro_short_sents.txt'


def tokenize_sentences(sentences):
    this_tk = Tokenizer()
    this_tk.fit_on_texts(sentences)
    this_sents = this_tk.texts_to_sequences(sentences)
    return this_sents, this_tk

def file_to_sents_list(filename: str) -> list:
    sents_list = []
    with open(filename,'r') as source:
        for line in source:
            if line.rstrip('\n') == '':
                print('empty line')
            else:
                sents_list.append(line.rstrip('\n'))
    #to limit training dataset size
    #return sents_list[:10000]
    return sents_list



def this_tokenize(sents):
    sents, tokenizer = tokenize_sentences(sents)
    return sents, tokenizer

l1_sents = file_to_sents_list(l1_file)
l2_sents = file_to_sents_list(l2_file)

l1_sents = l1_sents[:1000000]
l2_sents = l2_sents[:1000000]

l1_token_sents, l1_tokenizer = tokenize_sentences(l1_sents)
l2_token_sents, l2_tokenizer = tokenize_sentences(l2_sents)


l1_max = max([len(sent) for sent in l1_token_sents])
l2_max = max([len(sent) for sent in l2_token_sents])
this_max = max(l1_max, l2_max)
print(this_max)
cutoff = 20
cutoff_min = 5

l1_proc_token_sents = []
l1_proc_sents = []
l2_proc_token_sents = []
l2_proc_sents = []

num_good_sents = 0
for i in range(len(l1_sents)):
    this_sent_good = True
    if len(l1_token_sents[i]) > cutoff or len(l2_token_sents[i]) > cutoff:
        this_sent_good = False
    if len(l1_token_sents[i]) < cutoff_min or len(l2_token_sents[i]) < cutoff_min:
        this_sent_good = False
    
    #can add checking for infrequent terms here
    #########
    if this_sent_good:
        l1_proc_sents.append(l1_sents[i])
        l2_proc_sents.append(l2_sents[i])
        num_good_sents += 1
    if num_good_sents >= 200000:
        break

def printout(sents, filename):
    with open(filename, 'w') as sink:
        for sent in sents:
            sink.write(sent + '\n')
    return

printout(l1_proc_sents, l1_out)
printout(l2_proc_sents, l2_out)



