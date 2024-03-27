






langs = ['fr','it','pt','ro']


synt_tags = set()


for lang in langs:
    filename = 'MCCA_es-' + lang + '_' + lang + '_short_sents.txt.tags'
    with open(filename,'r') as source:
        for line in source:
            tag = line.rstrip('\n').lower()
            try:
                synt_tags.add(tag)
            except:
                pass
print(synt_tags)
