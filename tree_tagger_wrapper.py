#! /usr/bin/env python

#import subprocess
import os
from nltk.tokenize import word_tokenize
from datetime import datetime





#proc = os.system('echo \'hello world???\'')


test_sent = 'hi my name is nick'
it_sent = 'ciao mi chiamo nick e sono uno stupido'

def tag_sentence(sentence: str, parser_name: str):
    tmp_in_file = 'tmp_input.txt'
    tmp_out_file = 'tmp_output.txt'

    sent_toks = sentence.split()

    with open(tmp_in_file,'w') as sink:
        for word in sent_toks:
            sink.write(word + '\n')
    
    syscall = os.system('bin/tree-tagger lib/' + parser_name + '.par ' + tmp_in_file + ' ' + tmp_out_file)
    
    out_toks = []
    #out_sentence = '' 
    with open(tmp_out_file,'r') as sink:
        for line in sink:
            this_tok = line.rstrip('\n')
            out_toks.append(this_tok)
    out_sentence = ' '.join([tok for tok in out_toks])

    return out_sentence



def create_tags_doc(in_filename: str, out_filename: str, parser_name: str):
    tot_num_sents = 0
    with open(in_filename, 'r') as source:
        for count, line in enumerate(source):
            pass
        tot_num_sents = count + 1

    with open(in_filename,'r') as source:
        with open(out_filename,'w') as sink:
            for line in source:
                this_sent = line.rstrip('\n')
                this_tags = tag_sentence(this_sent, parser_name)
                sink.write(this_tags + '\n')
    return

def reformat_text_files(in_filename, out_filename):
    lines_per_sent_filename = in_filename + '.lps'
    with open(in_filename,'r') as source:
        with open(out_filename,'w') as sink:
            with open(lines_per_sent_filename,'w') as lps_sink:
                for line in source:
                    this_sent = word_tokenize(line)
                    num_toks_in_sent = len(this_sent)
                    tok_lines = '\n'.join([tok for tok in this_sent])
                    sink.write(tok_lines)
                    sink.write('\n\n')
                    lps_sink.write(str(num_toks_in_sent) + '\n')
    return lines_per_sent_filename


def create_tags_doc_v2(in_f, parser_name):
    
    out_f = in_f + '.tags'
    by_lines_f = in_f + '.by_lines'

    lps_f = reformat_text_files(in_f, by_lines_f)
    start = datetime.now()
    syscall = os.system('bin/tree-tagger lib/' + parser_name + '.par ' + by_lines_f + ' ' + out_f)
    end = datetime.now()
    print('it took this many seconds')
    print(str(end - start))
    return out_f

def reformat_tags_doc(tags_filename):
    out_filename = tags_filename + '.sent_per_line'
    with open(tags_filename,'r') as source:
        with open(out_filename,'w') as sink:
            for line in source:
                if line != 'FS\n':
                    sink.write(line.rstrip('\n') + ' ')
                elif line == 'FS\n':
                    sink.write(line)

def reformat_tags_doc_v2(by_lines_toks_filename, tags_filename):
    print('\n' + tags_filename)
    out_filename = tags_filename + '.sent_per_line'
    out_interleaved_filename = tags_filename + '.interleaved'
    #the tags from the one tag per line file
    tags_individ = []
    with open(tags_filename,'r') as tags_source:
        for line in tags_source:
            tags_individ.append(line.rstrip('\n'))
    #the tokens from the one token per line file
    toks_individ = []
    with open(by_lines_toks_filename,'r') as toks_source:
        for line in toks_source:
            if line != '\n':
                toks_individ.append(line.rstrip('\n'))
            else:
                #append an End of Sentence token
                toks_individ.append('<EOS>')
    #write to the reformatted only tags and to the reformatted interleaved
    with open(out_filename,'w') as tags_sink:
        with open(out_interleaved_filename,'w') as inter_sink:
            this_tags_sent = []
            this_interleaved_sent = []
            tot_toks = len(toks_individ)
            counter = 0
            for i in range(len(toks_individ)):
                this_tok = toks_individ[i]
                if this_tok != '<EOS>':
                    this_tag = tags_individ.pop(0)
                    this_tags_sent.append(this_tag)
                    this_interleaved_sent.append(this_tok)
                    this_interleaved_sent.append(this_tag)
                elif this_tok == '<EOS>':
                    #####end the sent, print it to the files
                    tags_sink.write(' '.join([tag for tag in this_tags_sent]) + '\n')
                    inter_sink.write(' '.join([tok for tok in this_interleaved_sent]) + '\n')
                    this_tags_sent = []
                    this_interleaved_sent = []
                counter += 1
                perc_done = 100.0 * float(counter) / float(tot_toks)
                print("{:.2f}".format(perc_done),end='\r')
    return



def generate_tags_wrap(filename, parser):
    tags_filename = create_tags_doc_v2(filename, parser)
    reformat_tags_doc(tags_filename)
    return

"""
reformat_text_files('test_doc.txt', 'test_reformat.txt')
print('reformatted')
create_tags_doc('test_doc.txt', 'test_tags.txt', 'italian')
test_tags = tag_sentence(it_sent, 'italian')
print(test_tags)
"""
#tags_filename = create_tags_doc_v2('es-it_es_short_sents.txt', 'spanish')
#reformat_tags_doc(tags_filename)

#init tag generation, but putting to line didn't work
"""
generate_tags_wrap('MCCA_es-fr_es_short_sents.txt', 'spanish')
generate_tags_wrap('MCCA_es-fr_fr_short_sents.txt', 'french')

generate_tags_wrap('MCCA_es-it_es_short_sents.txt', 'spanish')
generate_tags_wrap('MCCA_es-it_it_short_sents.txt', 'italian')

generate_tags_wrap('MCCA_es-pt_es_short_sents.txt', 'spanish')
generate_tags_wrap('MCCA_es-pt_pt_short_sents.txt', 'portuguese')

generate_tags_wrap('MCCA_es-ro_es_short_sents.txt', 'spanish')
generate_tags_wrap('MCCA_es-ro_ro_short_sents.txt', 'romanian')
"""

#retrying tag reformatting
reformat_tags_doc_v2('MCCA_es-fr_es_short_sents.txt.by_lines','MCCA_es-fr_es_short_sents.txt.tags')
reformat_tags_doc_v2('MCCA_es-fr_fr_short_sents.txt.by_lines','MCCA_es-fr_fr_short_sents.txt.tags')

reformat_tags_doc_v2('MCCA_es-it_es_short_sents.txt.by_lines','MCCA_es-it_es_short_sents.txt.tags')
reformat_tags_doc_v2('MCCA_es-it_it_short_sents.txt.by_lines','MCCA_es-it_it_short_sents.txt.tags')

reformat_tags_doc_v2('MCCA_es-pt_es_short_sents.txt.by_lines','MCCA_es-pt_es_short_sents.txt.tags')
reformat_tags_doc_v2('MCCA_es-pt_pt_short_sents.txt.by_lines','MCCA_es-pt_pt_short_sents.txt.tags')

reformat_tags_doc_v2('MCCA_es-ro_es_short_sents.txt.by_lines','MCCA_es-ro_es_short_sents.txt.tags')
reformat_tags_doc_v2('MCCA_es-ro_ro_short_sents.txt.by_lines','MCCA_es-ro_ro_short_sents.txt.tags')




