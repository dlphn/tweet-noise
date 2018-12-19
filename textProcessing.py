import spacy

nlp = spacy.load('fr')  # fr_core_news_sm
nlp2 = spacy.load('fr_core_news_md')

# Process whole documents
text = (u"Noël approche, les Français comme la plupart des Européens se "
        u"préparent à cette fête familiale, la plus importante de l’année. "
        u"En France, la plupart des gens fêtent Noël sauf évidemment les "
        u"pratiquants des autres religions. Vous devez savoir que le Père "
        u"Noël qui est aujourd’hui indissociable de Noël est arrivé en France "
        u"assez tard. On peut même dire qu’il est devenu de plus en plus "
        u"populaire quand les Français ont commencé à déserter les églises.")
doc = nlp(text)

# Tokenization

for token in doc:
    print(token.text)

print("======================================")

# Part-of-speech tagging
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

print("======================================")

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

print("======================================")

# Determine semantic similarities
doc1 = nlp(u"me frites étaient trop dégueu")
doc2 = nlp(u"des frites tellement dégoûtantantes")
similarity = doc1.similarity(doc2)
print(doc1.text, doc2.text, similarity)

