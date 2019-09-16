# Part-of-Speech (PoS) Tagging

This module aims to provide easier ways to work with PoS taggers. It extends some of the functionality provided by NLTK providing a map between MacMorpho and Floresta tagsets to 
the Universal tagset.

## Corpora

Currently this module supports three Portuguese corpora:

- [MacMorpho corpus](http://nilc.icmc.usp.br/macmorpho/) and [manual (in Portuguese)](http://nilc.icmc.usp.br/macmorpho/macmorpho-manual.pdf)
- [Floresta corpus](https://www.linguateca.pt/Floresta/) and [manual](http://visl.sdu.dk/visl/pt/symbolset-floresta.html)
- [LacioWeb corpus](http://www.nilc.icmc.usp.br/nilc/tools/nilctaggers.html) and [manual (in Portuguese)](http://www.nilc.icmc.usp.br/nilc/download/tagsetcompleto.doc)

NLTK already supports MacMorpho and Floresta corpora. LacioWeb is manually downloaded and parsed using NLTK's TaggedCorpusReader.

## Universal PoS Tagset

This module provide conversions from different tagsets to the Universal PoS Tagset proposed by Petrov, Das and McDonald. Link to paper: https://arxiv.org/abs/1104.2086

The original Universal PoS Tagset is composed of the following 12 tags:
    
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
