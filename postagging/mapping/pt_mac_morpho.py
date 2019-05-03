class MacMorpho:

    @classmethod
    def to_universal(self):
    '''
       Provide a mapping from Mac-Morpho tagset to the Universal
       Tagset. The Mac-Morpho tagset was obtained using a set to
       store only unique occurrences of tags.
       Mac-Morpho manual:
       http://nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf

       Complementary tags:

           - TAG|EST:      "Estrangeirismo";
           - TAG|AP:       apposition;
           - TAG|DAD:      data;
           - TAG|TEL:      phones;
           - TAG|DAT:      dates;
           - TAG|HOR:      hours;
           - TAG|+:        contractions and "ênclises";
           - TAG|!:        "mesóclise";
           - TAG|[TAG]:    discontinuity.

       
   '''
   d = {'!':       '.',
        '"':       '.',
        '$':       '.', # According to Wikipedia, this is a punctuation mark.
        "'":       '.',
        '(':       '.',
        ')':       '.',
        ',':       '.',
        '-':       '.',
        '.':       '.',
        '/':       '.',
        ':':       '.',
        ';':       '.',
        '?':       '.',
        'ADJ':     'ADJ',
        'ADJ|EST': 'ADJ',
        'ADV':     'ADV',
        'ADV-KS':  'ADV',
        'ADV-KS-REL': 'ADV',
        'ADV|+':   'ADV',
        'ADV|EST': 'ADV',
        'ADV|[':   'ADV',
        'ADV|]':   'ADV',
        'ART':     'ART',
        'ART|+':   'ART',
        'CUR':     'X',
        'IN':      'X',
        'KC':      'CONJ',
        'KC|[',    'CONJ',
        'KC|]':    'CONJ',
        'KS':      'CONJ',
        'N':       'NOUN',
        'NPRO':    'NOUN', # Manual does not report this tag. Probably a typo.
        'NPROP':   'NOUN',
        'NPROP|+': 'NOUN',
        'NUM':     'NUM',
        'NUM|TEL': 'NUM',
        'N|AP':    'NOUN',
        'N|DAT':   'NOUN',
        'N|EST':   'NOUN',
        'N|HOR':   'NOUN',
        'N|TEL':   'NOUN',
        'PCP':     'VERB', # Is this the participle?
        'PDEN':    'PRT',
        'PREP':    'ADP',
        'PREP|':   'ADP',
        'PREP|+':  'ADP',
        'PREP|[':  'ADP',
        'PREP|]':  'ADP',
        'PRO-KS':  'PRON',
        'PRO-KS-REL': 'PRON',
        'PROADJ':  'PRON',
        'PROPESS': 'PRON',
        'PROSUB':  'PRON',
        'V':       'VERB',
        'VAUX':    'VERB',
        'VAUX|+':  'VERB',
        'V|+':     'VERB',
        '[':       '.'}



