import spacy
from spacy import displacy

nlp = spacy.load('fr')  # fr_core_news_sm
nlp2 = spacy.load('fr_core_news_md')

text = (u"Noël approche, les Français comme la plupart des Européens se "
        u"préparent à cette fête familiale, la plus importante de l’année. "
        u"En France, la plupart des gens fêtent Noël sauf évidemment les "
        u"pratiquants des autres religions. Vous devez savoir que le Père "
        u"Noël qui est aujourd’hui indissociable de Noël est arrivé en France "
        u"assez tard. On peut même dire qu’il est devenu de plus en plus "
        u"populaire quand les Français ont commencé à déserter les églises.")

spam = (u"Ca va repartir sur du gros commit aujourd'hui, objectif niveau du 25 Nov,"
        u"Tmtc @Delphin71721317  et tmtc pas @vltnlfr")

info = (u"RCA : à l’ONU, dissensions toujours vives sur le rôle joué par la Russie https://t.co/kolIBp3Phz")

info2 = (u"Deux familles détournent plus d'1,7 million d'euros de prestations sociales en France")

# Tests
doc = nlp2(spam)

# Tokenization
for token in doc:
    print(token.text)

print("======================================")

# Visualization
# sentence_spans = list(doc.sents)
# displacy.serve(sentence_spans, style='dep')
# displacy.serve(doc, style='ent')

print("======================================")

# Part-of-speech tagging
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#           token.shape_, token.is_alpha, token.is_stop)
#
# print("======================================")

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

print("======================================")

# Determine semantic similarities
doc1 = nlp(u"mes frites étaient trop dégueu")
doc2 = nlp(u"des frites tellement dégoûtantes")
similarity = doc1.similarity(doc2)
print(doc1.text, doc2.text, similarity)

