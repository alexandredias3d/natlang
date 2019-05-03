
class UniversalTagset:
    '''
        Provide conversions from different tagsets to the Universal
        PoS Tagset proposed by Petrov, Das and McDonald. 
        Link to paper: https://arxiv.org/abs/1104.2086

        The Universal PoS Tagset is composed of the following 12 tags:
            
            - NOUN: nouns;
            - VERB: verbs;
            - ADJ:  adjectives;
            - ADV:  adverbs;
            - PRON: pronouns;
            - DET:  determiners and articles;
            - ADP:  prepositions and postpositions;
            - NUM:  numerals;
            - CONJ: conjunctions;
            - PRT:  particles;
            - .:    punctuation marks;
            - X:    everything else.

    '''
