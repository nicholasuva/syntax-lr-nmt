#! /usr/bin/env python

import os
import collections

import numpy as np
import tensorflow.compat.v1 as tf

import math

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from nltk.translate.bleu_score import sentence_bleu

#from tensorflow.compat.v1 import keras
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.compat.v1.keras.models import Model, Sequential, load_model
from tensorflow.compat.v1.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM, Masking
from tensorflow.compat.v1.keras.layers import Embedding
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, sparse_categorical_crossentropy, categorical_crossentropy#, softmax_crossentropy
#from tensorflow.compat.v1.keras.nlp.utils import beam_search


punct: set = {',', '.', '\'', '\"', ':', ';', '$', '#', '&', '-', '+', '=', '_', '?', '!'}
synt_tags: set = {'mcfprly', 'cscsp', 'va--2p', 'afp-p-ny', 'dw3mpr', 'vmp', 'pp2-pd--------s', 'afpmprn', 'px3--d--------w', 'ncfpry', 'mffsrln', 'pp3fpr--------s', 'sps+pp3', 'di3--r---e', 'abr', 'afpfsry', 'ver:futu', 'pon', 'ncmn', 'pd3fpo', 'vmsp3', 'pun:cit', 'ver:impe', 'pp2-pd--------w', 'pw3-so', 'aos', 'tdmsr', 'np0', 'ncmsvn', 'vmm-2p', 'pro:indef', 'pp1-pd--------w', 'pro:refl', 'vmii2p', 'sps', 'nc---n', 'ver:cimp', 'rgs', 'ds2fsrs', 'yp', 'tifso', 'sp+pd', 'ds1ms-p', 'di3fso', 'pd3msr', 'ds1fp-p', 'ls', 'di3fpr---e', 'mc-p-l', 'ds2ms-p', 'tf-so', 'vmsp3s', 'afpmsry', 'ver:pres', 'afcfsrn', 'afpfprn', 'vmn', 'vmp--sm---y', 'ynmsry', 'ti-po', 'pp3fpa--------w', 'pro:pos', 'comma', 'vmp--sm', 'ncfsryy', 'pro:rel', 'afpmpvy', 'vmsp2s', 'vmis2s', 'ynfsry', 'ncmpry', 'pp2-sn--------s', 'pp1-pa--------w', 'afpmp-n', 'mofpoly', 'npmpoy', 'pre:det', 'vmip1p', 'fla', 'vmg', 'fe', 'npmsoy', 'qn', 'ds2fp-s', 'vmm-2s', 'tdmso', 'ver:remo', 'fh', 'ncfp', 'aqs', 'y', 'afpmsoy', 'pp2-----------s', 'npmsry', 'ds3fsos', 'px3', 'det:indef', 'npmprn', 'vmm', 'afp-p-n', 'ynmpry', 'ver:infi', 'mlmpr', 'di3-po', 'dh2ms', 'vmm-2s----y', 'mcmp-l', 'ncfs-n', 'ncms-ny', 'di3-po---e', 'npfsoy', 'dh3fsr', 'aq0', 'afpms-n', 'fg', 'rp', 'pp1-pa--y-----w', 'px3--a--------w', 'ccssp', 'ncfpon', 'ver:cpre', 'pro:per', 'pro:pers', 'di3mpr---e', 'vmi', 'pz3fsr', 'v+pp', 'afpmpoy', 'pp1-pd--------s', 'vmil2s', 'ds2ms-s', 'colon', 'tdfsr', 'aqd', 'ps2mp-s', 'npmsvn', 'dh3fp', 'adv', 'ncfsoy', 'pp2-pa--------w', 'pp2-pd--y-----w', 'dd3msr---o', 'pp1-sa--------w', 'afpmson', 'vmsp1p', 'vmii1-----y', 'td-po', 'dd3msr---e', 'ds3ms-s', 'px2', 'pp1-sd--------s', 'ts-po', 'afsfp-n', 'pp2-sd--------w', 'dd3fso---o', 'mcmsrl', 'dh1fsr', 'dd3fso', 'npr', 'ccsspy', 'npfsrn', 'ncmpoy', 'dd3mpr---o', 'vmip1s----y', 'pd0', 'ver:subp', 'ver:impf', 'mlmpo', 'afsms-n', 'mffsrly', 'fc', 'pi3-so', 'di3', 'ps1fsrs', 'mcmsoly', 'vmip3s', 'slash', 'tifsr', 'rw', 'ncmprn', 'mcfpoly', 'fd', 'npfsry', 'cs', 'ord', 'afpmpry', 'mcms-ln', 'ds2mp-s', 'mlfpr', 'ncmsrn', 'ynfsoy', 'vmsp3-----y', 'ncfp-ny', 'pro:dem', 'timsr', 'ncmp-n', 'pp3fsa--y-----w', 'ds1fsrp', 'con', 'yr', 'ds2mp-p', 'fp', 'fat', 'mo-s-r', 'hyphen', 'dd3fsr', 'vmip2s----y', 'dd3mso---e', 'pz3msr', 'kon', 'vmnp', 'det:pos', 'fx', 'pw3mso', 'di3fso---e', 'vmis3p', 'ncfpryy', 'ncmpvy', 'ps1ms-p', 'afpmsrn', 'afsmp-n', 'mofsoly', 'pp3msa--y-----w', 'vmil3p', 'ds1fsrs', 'pd3mpr', 'sym', 'dz3fsr---e', 'mofprly', 'pd3mpo', 'ds3mp-s', 'momsrly', 'va--1s', 'vmmp2s', 'ds2fsos', 'vmip2p', 'ncfpoy', 'ds1mp-p', 'pi3--r', 'di3-sr---e', 'afpfsrn', 'pw3--r', 'mofsrly', 'paren', 'ver:cond', 'pw3fpr', 'pp2-sd--------s', 'sent', 'vmis3s----y', 'dh1ms', 'ncfsrn', 'prp', 'afpf--n', 'qf', 'pp3', 'vmip1s', 'pp3mpr--------s', 'dz3mso', 'moms-l', 'dd3fpr---e', 'va--2s', 'pro:poss', 'rgp', 'rgpy', 'fpa', 'dh1mp', 'vms', 'mcfsrln', 'tdfso', 'vmil1p', 'yp-p', 'afpfp-n', 'sp+da', 'pd3fpr', 'tsfs', 'ml-po', 'ver:pper', 'pp3msr--------s', 'afpmpon', 'pro:ind', 'ver:simp', 'va--1p', 'det:def', 'pro:demo', 'fpt', 'mc-s-d', 'aqc', 'tsfp', 'pi3-sr', 'dh3mp', 'pp3mpa--------w', 'dd3fpr', 'pp3fsr--y-----s', 'ps1fsrp', 'pw3fso', 'dz3fso', 'dw3-po---e', 'ds1fp-s', 'vmii3s----y', 'pd3mso', 'afpfson', 'dd3fsr---o', 'fca', 'fw', 'vmip3s----y', 'pp1-sa--------s', 'vmip3', 'vmip1p----y', 'sps+sps', 'ncmsry', 'prp:det', 'afpfpry', 'spsa', 'dw3fpr', 'mcfprln', 'dd3fpo', 'mcmsrly', 'dd3fso---e', 'ao0', 'sps+rg', 'afpfsvy', 'dd3mpr', 'afpfsoy', 'ncfs', 'pi3fso', 'pp2-sa--------w', 'vmil3s', 'pz3-so', 'pw3mpr', 'ver:subi', 'di3mso---e', 'ncfp-n', 'mcfp-l', 'di3-sr', 'rg', 'qs', 'rc', 'ncmsoy', 'pd3-po', 'ncmson', 'pro:inter', 'pr0', 'mlfpo', 'np', 'fct', 'ds2fsrp', 'mcfsoln', 'pi3msr', 'mcfp-ln', 'mofs-l', 'va--3p', 'ncfsrny', 'di3ms', 'afcfsry', 'afpfsrny', 'fs', 'vmis1s', 'ncfpvy', 'ds3fp-s', 'mofp-ln', 'ver:refl:infi', 'pp1-pr--------s', 'afcfson', 'z', 'pd3fsr', 'pp1-sr--------s', 'det:art', 'pp3-po--------s', 'dd3mpr---e', 'dd3-po---o', 'afsfsrn', 'pp2', 'vmnp------y', 'pw3msr', 'pun', 'flt', 'quote', 'vmsp2p', 'sps+di0', 'ver:geru', 'px3--a--------s', 'ft', 'ncfsry', 'adj', 'pp3fso--------s', 'moms-ln', 'di3fpr', 'dp2', 'dp3', 'x', 'npfp-n', 'nccp', 'tffpry', 'pp1', 'rz', 'vmil1s', 'rn', 'pw3-po', 'pp3-sd--y-----w', 'ncms', 'ncms-n', 'nom', 'pp2-pr--------s', 'spsg', 'ds3fsrs', 'mofsrln', 'pp3-pd--------w', 'dd3mso---o', 'vmsp1s', 'fz', 'crssp', 'dw3--r---e', 'csssp', 'ncm--n', 'di3fsr---e', 'pp2-sa--------s', 'pp3fsr--------s', 'ds1fsop', 'pro:rela', 'dd3msr', 'afcmpry', 'dp1', 'dw3fsr', 'ps2ms-s', 'tsms', 'pp2-sr--------s', 'yn', 'pi0', 'afpm--n', 'momprly', 'npfson', 'fit', 'ncmsvy', 'afcmp-n', 'afp', 'pz3-sr', 'pp3mso--------s', 'px1', 'cc', 'dd3fpr---o', 'ds2fp-p', 'pe0', 'vmis1p', 'di3msr', 'da0', 'ncfn', 'tsmp', 'va--1', 'pp2-pa--y-----w', 'mffprln', 'vmii2s', 'pp+pp', 'qz', 'tfmsry', 'di3fsr', 'spsay', 'di3fp', 'afcfp-n', 'mompoly', 'pi3mpr', 'pre', 'tdmpr', 'dd3fsr---e', 'mo---l', 'afcms-n', 'cssspy', 'afpfp-ny', 'ver:ppre', 'pi3fsr', 'afpms-ny', 'ynmsvy', 'pp1-sn--------s', 'va--3', 'nam', 'nccn', 'afpmsvy', 'di3mp', 'pp1-sd--y-----w', 'dd3fpr--y', 'ncfprn', 'int', 'dz3msr---e', 'dd0', 'mo---ln', 'px3--d--y-----w', 'dh3ms', 'vmip3p', 'dh2mp', 'spsd', 'afs', 'ps2fsrs', 'dw3msr', 'ncmsryy', 'pt0', 'ncfsvy', 'pi3msr--y', 'pi3mso', 'num', 'di3msr---e', 'ps3mp-s', 'afpfpoy', 'pi3fpr', 'ds1mp-s', 'nccs', 'vmii3p', 'vmg-------y', 'pp3msa--------w', 'va--3s----y', 'afpmp-ny', 'vmip2s', 'dd3-po---e', 'sp+dd', 'ncfson', 'vmis3s', 'di3mpr', 'pi3-po', 'pro', 'tdfpr', 'vmii3s', 'ncmpon', 'vanp', 'ncf--n', 'ncmp', 'pp3fsa--------w', 'vmii1', 'spca', 'qz-y', 'i', 'pp1-sd--------w', 'momsoly', 'afpfpon', 'va--3s', 'pp3-sd--------w', 'ds1ms-s', 'di0', 'timso'}




def tokenize_sentences(sentences: list):
    this_tokenizer = Tokenizer(
        #filters='"#$%&()*+-/<=>@[\\]^_`{|}~\t\n',
        lower=True
    )
    this_tokenizer.fit_on_texts(sentences)
    this_sentences = this_tokenizer.texts_to_sequences(sentences)
    return this_sentences, this_tokenizer

#es_filename = 'DGT.es-pt.es'
#pt_filename = 'DGT.es-pt.pt'

#es_filename = 'DGT_plaintext_es-fr.txt/DGT.es-fr.es'
#pt_filename = 'DGT_plaintext_es-fr.txt/DGT.es-fr.fr'

#pt_filename = 'small_vocab_en.txt'
#es_filename = 'small_vocab_fr.txt'

#es_filename = 'es_short_sents.txt'
#pt_filename = 'pt_short_sents.txt'

#es_filename = 'es-fr_es_short_sents.txt'
#pt_filename = 'es-fr_fr_short_sents.txt'

#es_filename = 'News-Commentary.es-it.es'
#pt_filename = 'News-Commentary.es-it.it'

#es_filename = 'es-it_es_short_sents.txt'
#pt_filename = 'es-it_it_short_sents.txt'

#es_filename = 'MCCA_es-pt_es_short_sents.txt.tags.interleaved'
#pt_filename = 'MCCA_es-pt_pt_short_sents.txt.tags.interleaved'


es_filename = 'MCCA_es-ro_es_short_sents.txt'
pt_filename = 'MCCA_es-ro_ro_short_sents.txt'

"""
es_filename = 'MCCA_es-pt_es_short_sents.txt'
pt_filename = 'MCCA_es-pt_pt_short_sents.txt'
es_tags_filename = 'MCCA_es-pt_es_short_sents.txt.tags.sent_per_line'
pt_tags_filename = 'MCCA_es-pt_pt_short_sents.txt.tags.sent_per_line'
"""
"""
es_filename = 'MCCA_es-fr_es_short_sents.txt.tags.interleaved'
pt_filename = 'MCCA_es-fr_fr_short_sents.txt.tags.interleaved'
"""
t_s_id = 5

def file_to_sents_list(filename: str) -> list:
    sents_list = []
    with open(filename,'r') as source:
        for line in source:
            if line.rstrip('\n') == '':
                pass
                #print('empty line')
            else:
                sents_list.append(line.rstrip('\n'))
    #to limit training dataset size
    #sents_list.extend(sents_list)
    #sents_list.extend(sents_list)
    #sents_list.extend(sents_list)
    return sents_list[:51000]
    #return sents_list

es_sents = file_to_sents_list(es_filename)
pt_sents = file_to_sents_list(pt_filename)
#es_tags = file_to_sents_list(es_tags_filename)
#pt_tags = file_to_sents_list(pt_tags_filename)


def interleave_lists(sents_list: list, tags_list: list):
    interleaved_list = []
    for i in range(len(sents_list)):
        this_sent_toks = sents_list[i].split(' ')
        this_sent_tags = tags_list[i].split(' ')
        this_interleaved = []
        for i in range(len(this_sent_toks)):
            try:
                this_interleaved.append(this_sent_toks[i])
                this_interleaved.append(this_sent_tags[i])
            except IndexError:
                pass
        interleaved_sent = ' '.join([word for word in this_interleaved])
        interleaved_list.append(interleaved_sent)
    return interleaved_sent



def pad(sentences, max_length):
    return pad_sequences(sentences, maxlen = max_length, padding = 'post')
     
def preprocess_parallel_texts(x,y):
    #print('first sentence before preproc-----')
    #print(x[t_s_id])
    
    #Try adding an <END> of sentence tag
    """
    for i in range(len(x)):
        x[i] = x[i] + ' <END>'
        y[i] = y[i] + ' <END>'
    """

    x_sents, x_tokenizer = tokenize_sentences(x)
    y_sents, y_tokenizer = tokenize_sentences(y)

    #print('1st sent tokenized----')
    #print(x_sents[t_s_id])
    max_x = max([len(sentence) for sentence in x_sents]) 
    max_y = max([len(sentence) for sentence in y_sents])
    """
    I need to check that I'm not getting any 0 length sentences, and remove those
    sentences from both languages
    """

    max_length = max(max_x, max_y)
    x_sents = pad(x_sents, max_length)
    y_sents = pad(y_sents, max_length)

    #print('1st sent padded ---------')
    #print(x_sents[t_s_id])

    #x_sents = x_sents.reshape(*x_sents.shape, 1)
    #y_sents = y_sents.reshape(*y_sents.shape, 1)

    #trying this way from stackoverflow lol
    """
    x_sents_reshaped = x_sents.reshape(x_sents.shape[0], 1, x_sents.shape[1])
    y_sents_reshaped = y_sents.reshape(y_sents.shape[0], 1, y_sents.shape[1])
    """

    #refactoring lol
    """
    x_sents_reshaped = x_sents.reshape(x_sents.shape[0], x_sents.shape[1], 1)
    y_sents_reshaped = y_sents.reshape(y_sents.shape[0], y_sents.shape[1], 1)
    """

    #for v2
    x_sents_reshaped = x_sents.reshape(x_sents.shape[0], x_sents.shape[1], 1)
    y_sents_reshaped = y_sents.reshape(y_sents.shape[0], y_sents.shape[1], 1)
    return x_sents_reshaped, y_sents_reshaped, x_tokenizer, y_tokenizer


def build_encoder_decoder(this_input_shape, output_sequence_length, l1_vocab_size, l2_vocab_size):
    learning_rate = 0.005
    #learning_rate = 1e-3
    this_batch_size = 128
    model = Sequential()
    #i think not having masking layer was the prob, but not sure about input shape on this masking layer, I'm assuming it should be the same as the data?
    #model.add(Masking(mask_value = 0, input_shape = this_input_shape[1:]))
    #model.add(Masking(mask_value = 0))
    #mask_shape = [this_batch_size].append(this_input_shape[1:])
    #this is an awful solution
    #mask_shape = np.empty([this_batch_size, this_input_shape[1], this_input_shape[2]]).shape
    mask_shape = this_input_shape[1:]
    model.add(Masking(mask_value = 0, input_shape = mask_shape))
    model.add(GRU(this_batch_size, input_shape = this_input_shape[1:], return_sequences = False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(this_batch_size, return_sequences = True))
    model.add(TimeDistributed(Dense(l2_vocab_size, activation = 'softmax')))
    model.compile(loss = sparse_categorical_crossentropy, optimizer = Adam(learning_rate), metrics = ['accuracy'])
    return model

def build_encoder_decoder_v2(this_input_shape, output_sequence_length, l1_vocab_size, l2_vocab_size):
    learning_rate = 0.0005
    #learning_rate = 1e-3
    #this_batch_size = 256
    model = Sequential()
    mask_shape = this_input_shape[1:]
    print('model shape' + str(this_input_shape))
    print(str(this_input_shape[1:]))
    
    model.add(Masking(mask_value = 0, input_shape = mask_shape))
    #model.add(Embedding(input_dim=l1_vocab_size, output_dim=128, input_length = this_input_shape[1]))
    model.add(Bidirectional(GRU(256, input_shape = this_input_shape[1:], return_sequences = False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256, return_sequences = True)))
    model.add(TimeDistributed(Dense(l2_vocab_size, activation = 'softmax'))) 
    acc = 'accuracy'
    #acc = tf.keras.metrics.SparseCategoricalAccuracy()
    #model.compile(loss = sparse_categorical_crossentropy, sample_weight_mode='temporal', optimizer = Adam(learning_rate), metrics = ['accuracy'])
    this_loss = SparseCategoricalCrossentropy(from_logits=False)
    model.compile(loss = this_loss, sample_weight_mode='temporal', optimizer = Adam(learning_rate), metrics = [acc])
    return model

def bleu_calc(model, src_sents_vec, ref_sents):
    tot_vec = float(len(src_sents_vec))
    predictions = []
    ct = 0.0
    for vec in src_sents_vec:
        thingy = np.array([vec])
        pred = model.predict(thingy)
        predictions.append(pred[0])
        ct += 1.0
        pct = ct / tot_vec
        print(f'{pct:.2f}', end='\r')
    #predictions = model.predict(ref_sents_vec)
    pred_sents = []
    
    #ref_sents = []
    for pred in predictions:
        #print(pred)
        this_sent = text_retrieve(pred, l2_tokenizer)
        pred_sents.append(this_sent)
    ref_filename = 'pt_ref_sents.txt'
    pred_filename = 'pt_hyp_sents.txt'
    with open(ref_filename, 'w') as sink:
        for ref in ref_sents:
            sink.write(ref.lower() + '\n')
            #sink.write(' '.join([word for word in ref]))
    with open(pred_filename ,'w') as sink:
        for pred in pred_sents:
            sink.write(pred + '\n')
            #sink.write(' '.join([word for word in pred]))
    
    tot_sents = len(ref_sents)
    #calc bleu
    bleu_sum = 0.0
    bleu_max = 0.0
    for i in range(tot_sents):
        ref_tmp = ref_sents[i].lower().split()
        ref = [[]]
        for tok in ref_tmp:
            if tok not in synt_tags and tok not in punct:
                ref[0].append(tok)
        pred = pred_sents[i].split()
        this_bleu = sentence_bleu(ref, pred, weights=(1.0, 0.0, 0.0, 0.0))
        #this_bleu = sentence_bleu(ref, pred, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_sum += this_bleu
        if i < 10:
            print(ref)
            print(pred)
            print(this_bleu)
        if this_bleu > bleu_max:
            bleu_max = this_bleu
    bleu = bleu_sum / float(tot_sents)
    print('bleu_max:\t' + str(bleu_max))
    test_ref = [['this', 'is', 'a', 'test']]
    test_hyp = ['this', 'is','the', 'test']
    #print('testing implementation:')
    #print(sentence_bleu(test_ref, test_hyp, weights=(0.25, 0.25, 0.25, 0.25)))
    return bleu


l1_sentences, l2_sentences, l1_tokenizer, l2_tokenizer = preprocess_parallel_texts(es_sents, pt_sents)
#l1_tags, l2_tags, l1_tag_tokenizer, l2_tag_tokenizer = preprocess_parallel_texts(es_tags, pt_tags)

#try adding the dimensionality of the tags to the data
def add_tag_feats(sentences, tags):
    x = sentences.shape[0]
    y = sentences.shape[1]
    print(sentences.shape)
    print(tags.shape)
    tagged_sentences = np.zeros((x,y,2)) #.reshape(x,y,2)
    #tagged_sentences = sentences
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            this_word_ind = sentences[i,j,0]
            this_tag_ind = tags[i,j,0]
            #tagged_sentences[i,j,1] = this_tag_ind
            tagged_sentences[i,j] = np.array([this_word_ind,this_tag_ind])
    print(tagged_sentences.shape)
    return tagged_sentences

"""
l1_sentences = add_tag_feats(l1_sentences, l1_tags)
l2_sentences = add_tag_feats(l2_sentences, l2_tags)
"""

tmp_max = max([len(sentence) for sentence in l1_sentences])

#try generating class weight?
#------------------
#class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=l2_sentences)
#class_weight_dict = dict(enumerate(class_weights))
word2freq = l2_tokenizer.word_counts
word2id = l2_tokenizer.word_index
total_toks = 0
for key, val in word2freq.items():
    total_toks += val
class_weights = {}
testi = 0
for key, val in word2id.items():
    #log smoothing bad
    tmp_weight = 1.0 / float(word2freq[key])
    #sigmoid smoothing
    this_weight = 1.0 / (1.0 + pow(math.e, ((-200*tmp_weight) +5.0)))
    class_weights[val] = this_weight
    """
    if testi < 2000:
        print(key)
        print(val)
        print(tmp_weight)
        print(this_weight)
        testi += 1
    """



#test train
l1_test = l1_sentences[50001:]
l2_test = l2_sentences[50001:]
l1_sentences = l1_sentences[:50000]
l2_sentences = l2_sentences[:50000]


sample_weights = np.zeros(l2_sentences.shape[:2])
print(sample_weights.shape)
print(l2_sentences.shape)
for i in range(len(l2_sentences)):
    for j in range(len(l2_sentences[i])):
        try:
            sample_weights[i,j] = 1.0
            #sample_weights[i,j] = class_weights[l2_sentences[i,j,0]]
        except KeyError:
            sample_weights[i,j] = 1.0
            #sample_weights[i,j] = 0.0000001


with tf.device('/DML:0'):
    """uncomment this to retrain ///////
    this_encoder_decoder = build_encoder_decoder_v2(
            l1_sentences.shape,
            l2_sentences.shape[1],
            len(l1_tokenizer.word_index)+1,
            len(l2_tokenizer.word_index)+1
            )
    this_encoder_decoder.fit(l1_sentences, l2_sentences, batch_size=54, epochs=20, validation_split=0.2, sample_weight=sample_weights)
    this_encoder_decoder.save('baseline_test_model/12_24_es_to_ro_interleaved_ttsplit_MCCA_lr0.0005_50k_GRU256_bat64_epo20_gpu_no_sub_5len_sents.pd')
    """

    this_encoder_decoder = load_model('baseline_test_model/12_23_es_to_ro baseline_ttsplit__MCCA_lr0.0005_50k_GRU256_bat64_epo20_gpu_no_sub_5len_sents.pd')
    #this_encoder_decoder = load_model('baseline_test_model/12_23_es_to_pt interleaved_ttsplit__MCCA_lr0.0005_50k_GRU256_bat64_epo20_gpu_no_sub_5len_sents.pd')
    #this_encoder_decoder = load_model('baseline_test_model/12_23_es_to_fr_interleaved_ttsplit_MCCA_lr0.0005_50k_GRU256_bat64_epo20_gpu_no_sub_5len_sents.pd')
    #this_encoder_decoder = load_model('baseline_test_model/12_23_es_to_pt interleaved_ttsplit__MCCA_lr0.0005_50k_GRU256_bat64_epo20_gpu_no_sub_5len_sents.pd')
    #this_encoder_decoder = load_model('baseline_test_model/12_23_es_to_it_baseline_MCCA_lr0.0005_50k_GRU256_bat128_epo20_gpu_no_sub_5len_sents.pd')
    #this_encoder_decoder.save('baseline_test_model/bltm.pd')
    #this_encoder_decoder = load_model('baseline_test_model/bltm.pd')
    #this_encoder_decoder = load_model('baseline_test_model/es_to_pt_150k_GRU256_bat256_epo5_gpu_no_sub_5len_sents.pd')
    #this_encoder_decoder = load_model('baseline_test_model/12_22_es_to_pt_DGT_150k_GRU256_bat256_epo5_gpu_no_sub_5len_sents.pd')
    #this_encoder_decoder = load_model('baseline_test_model/es_to_fr_DGT_150k_GRU256_bat256_epo5_gpu_no_sub_5len_sents.pd')
 

    #this_encoder_decoder = load_model('baseline_test_model/12_22_es_to_it_NC_lr0.001_lessthan150k_GRU256_bat256_epo1_gpu_no_sub_5len_sents.pd')





def token_prob_fn(inputs):
    return this_encoder_decoder(inputs)[:,-1,:]

def text_retrieve(data, tokenizer):
    #method to remove syntactic tags, this is not perfect, I know it removes NOM, need to fix
    #would prob require re-interlaving, like text->lower, then interleave and leave tags capital


    #{'prp:det', 'nom', 'adj', 'pun', 'det:art', 'adv', 'vlfin', 'art', 'nc', 'prep', 'np', 'cm', 'fs', 'prep', 'rel', 'se', 'card', 'csubi', 'cque', 'qu', 'dm', 'ppx', 'pdel', 'colon', 'alfs', 'neg', 'vladj', 'pe', 'ppo', 'semicolon'}
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    #for ind in index_to_words:
        #print(str(ind) + '\t' + str(index_to_words[ind]))
    index_to_words[0] = '<PAD>'
    #to not show padding char at all
    #for line in data:
        #line[0] = 0.0
    this_sent = []
    #the normal argmax retrieval of the prediction
    maxes = np.argmax(data, 1)
    #trying to add randomness
    #maxes = [np.random.choice(len(sent),p=sent) for sent in data]
    
    #trying beam search
    """
    prompt = tf.fill((10,1),1)
    maxes = beam_search(
            token_prob_fn,
            prompt,
            max_length=l2_sentences[1],
            num_beams=5
    )
    """

    for pred in maxes:
        #print('pred: ' + str(pred))
        if pred != 0:
            this_word = index_to_words[pred]
            if this_word not in synt_tags:
                this_sent.append(index_to_words[pred])
    return ' '.join([word for word in this_sent])
    
    #return ' '.join([index_to_words[prediction] for prediction in np.argmax(data, 1)])












with tf.device('/DML:0'):
    print("is gpu available:")
    print(tf.test.is_gpu_available())
    #print(tf.config.list_physical_devices('GPU'))

    #test_pred_set = l1_sentences[:20]
    #prediction = this_encoder_decoder.predict(test_pred_set)
    '''
    for elem in prediction:
        print(elem)
        print(text_retrieve(elem, l2_tokenizer))
    '''
    bleu_score = bleu_calc(this_encoder_decoder, l1_test, pt_sents[50001:])
    print('BLEU SCORE:')
    print(bleu_score)
    
