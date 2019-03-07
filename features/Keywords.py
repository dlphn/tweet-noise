keywords_blacklist = [
    # Mots Spam Arnaques & Bots
    '100% gratuit', 'acces gratuit', 'agissez des maintenant', 'annulation a tout moment', 'annulez a tout moment',
    'aucun cout', 'aucun engagement', 'aucun risque', 'bon plan', 'bonne affaire', 'cadeau',
    'carte bancaire', 'carte de credit', 'cartes acceptees', 'cb', 'certifie', 'cheque', 'cliquez', 'cliquez ici',
    'commandez aujourd’hui', 'commandez maintenant', 'devenez membre', 'duree limite', 'echantillon gratuit',
    'escroquerie', 'essai gratuit', 'facture', 'felicitations', 'free', 'gagnez', 'gratuit', 'incroyable',
    'inscrivez-vous gratuitement aujourd’hui', 'installation gratuite', 'interet', 'meilleur prix',
    'nouveaux clients uniquement', 'obtenez-le maintenant', 'offre exclusive', 'opportunite', 'pas de frais',
    'paypal', 'postulez', 'pour seulement', 'prix les plus bas', "profitez aujourd'hui", 'promotion speciale',
    'quantites limitees', 'réduction', 'sans engagement', 'sans frais', 'spam', "taux d'interet", 'temps limite',
    'unique',

    # Mots Spam corps et sexualité
    'bite', 'celibataire', 'chaud', 'cul', 'cure', 'fesse', 'hormone', 'hot', 'maigrir', 'perdre du poids', 'penis',
    'performance', 'regime', 'rides', 'sexe', 'sexuel', 'sexy', 'viagra',

    #Mots Spam Jeux d'argent
    'blackjack', 'casino', 'jetons', 'poker', 'roulette', 'parier', 'bet',

    #Mots Spam Twitter
    'hashtag', 'tweet',

    #Mots Spam Pub
    'concours', 'euros', 'jeu', 'maintenant', 'offre exclusive', 'promotion', "seulement aujourd'hui",

    #Mots Spam conversations privées
    'ahah', 'bisous', 'buzzword', 'chiant', 'cool', 'dab', 'galere', 'gueule', 'haha', 'hahaha', 'hahahaha',
    'hahahahaha', 'hihi', 'lol', 'maj', 'mdr', 'mec', 'meuf', 'nana', 'ok', 'oups', 'ptdr', 'relou', 'srab',
    'wallah', 'wesh', 'xd', 'yes', 'yolo'
                 ]



keywords_whitelist = ["news", "info", "actu", "actualite"]

keywords_whitelist_freq = ['National', 'condamné', 'Russie', 'enquête', 'Masson', 'Var', 'régime', 'risques', 'Davis',
                           'socialmedia', 'coalition', 'Bleus', 'Didier', 'Football', 'attendent', 'européens', 'juges',
                           'défis', 'donnait', 'enjeux', 'atmosphère', 'Finance', 'collectivités', 'Marine', 'Dhabi',
                           '3WAcademy', 'Nicolas', 'Actualité', 'lancé', 'Champ', 'Nouvelles', 'pesticides', 'casseurs',
                           'manifestent', 'impôts', 'locales', 'européenne', '“gilets', 'jaunes”', 'loi', 'manquez',
                           'marché', 'alerte', 'Mercato', 'trêve', 'sécession', 'élites', 'protéger', 'Gard', 'Yves',
                           'française', 'jaunes»', 'INFO', 'FRANCEINFO', 'mobilisation', 'reporte', 'vignette',
                           'Europe1', 'Force', '➡️', 'Sondage', 'soutiennent', 'milliards', 'CRS', 'Vanity', 'Fair',
                           'Planet', 'détenu', 'Agnès', 'lavoixdunord', 'élus']



keywords_stoplist = ["a", "abord", "absolument", "afin", "ah", "ai", "aie", "aient", "aies", "ailleurs", "ainsi", "ait",
                     "allaient", "allo", "allons", "allô", "alors", "anterieur", "anterieure", "anterieures", "apres",
                     "après", "as", "assez", "attendu", "au", "aucun", "aucune", "aucuns", "aujourd", "aujourd'hui",
                     "aupres", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "auriez",
                     "aurions", "aurons", "auront", "aussi", "autre", "autrefois", "autrement", "autres", "autrui",
                     "aux", "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avez", "aviez",
                     "avions", "avoir", "avons", "ayant", "ayez", "ayons", "b", "bah", "bas", "basee", "bat", "beau",
                     "beaucoup", "bien", "bigre", "bon", "boum", "bravo", "brrr", "c", "car", "ce", "ceci", "cela",
                     "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci",
                     "celui-là", "celà", "cent", "cependant", "certain", "certaine", "certaines", "certains", "certes",
                     "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque", "cher",
                     "chers", "chez", "chiche", "chut", "chère", "chères", "ci", "cinq", "cinquantaine", "cinquante",
                     "cinquantième", "cinquième", "clac", "clic", "combien", "comme", "comment", "comparable",
                     "comparables", "compris", "concernant", "contre", "couic", "crac", "d", "da", "dans", "de",
                     "debout", "dedans", "dehors", "deja", "delà", "depuis", "dernier", "derniere", "derriere",
                     "derrière", "des", "desormais", "desquelles", "desquels", "dessous", "dessus", "deux", "deuxième",
                     "deuxièmement", "devant", "devers", "devra", "devrait", "different", "differentes", "differents",
                     "différent", "différente", "différentes", "différents", "dire", "directe", "directement", "dit",
                     "dite", "dits", "divers", "diverse", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept",
                     "dixième", "doit", "doivent", "donc", "dont", "dos", "douze", "douzième", "dring", "droite", "du",
                     "duquel", "durant", "dès", "début", "désormais", "e", "effet", "egale", "egalement", "egales",
                     "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin", "entre", "envers",
                     "environ", "es", "essai", "est", "et", "etant", "etc", "etre", "eu", "eue", "eues", "euh",
                     "eurent", "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eux-mêmes",
                     "exactement", "excepté", "extenso", "exterieur", "eûmes", "eût", "eûtes", "f", "fais", "faisaient",
                     "faisant", "fait", "faites", "façon", "feront", "fi", "flac", "floc", "fois", "font", "force",
                     "furent", "fus", "fusse", "fussent", "fusses", "fussiez", "fussions", "fut", "fûmes", "fût",
                     "fûtes", "g", "gens", "h", "ha", "haut", "hein", "hem", "hep", "hi", "ho", "holà", "hop", "hormis",
                     "hors", "hou", "houp", "hue", "hui", "huit", "huitième", "hum", "hurrah", "hé", "hélas", "i",
                     "ici", "il", "ils", "importe", "j", "je", "jusqu", "jusque", "juste", "k", "l", "la", "laisser",
                     "laquelle", "las", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "longtemps",
                     "lors", "lorsque", "lui", "lui-meme", "lui-même", "là", "lès", "m", "ma", "maint", "maintenant",
                     "mais", "malgre", "malgré", "maximale", "me", "meme", "memes", "merci", "mes", "mien", "mienne",
                     "miennes", "miens", "mille", "mince", "mine", "minimale", "moi", "moi-meme", "moi-même",
                     "moindres", "moins", "mon", "mot", "moyennant", "multiple", "multiples", "même", "mêmes", "n",
                     "na", "naturel", "naturelle", "naturelles", "ne", "neanmoins", "necessaire", "necessairement",
                     "neuf", "neuvième", "ni", "nombreuses", "nombreux", "nommés", "non", "nos", "notamment", "notre",
                     "nous", "nous-mêmes", "nouveau", "nouveaux", "nul", "néanmoins", "nôtre", "nôtres", "o", "oh",
                     "ohé", "ollé", "olé", "on", "ont", "onze", "onzième", "ore", "ou", "ouf", "ouias", "oust", "ouste",
                     "outre", "ouvert", "ouverte", "ouverts", "o|", "où", "p", "paf", "pan", "par", "parce", "parfois",
                     "parle", "parlent", "parler", "parmi", "parole", "parseme", "partant", "particulier",
                     "particulière", "particulièrement", "pas", "passé", "pendant", "pense", "permet", "personne",
                     "personnes", "peu", "peut", "peuvent", "peux", "pff", "pfft", "pfut", "pif", "pire", "pièce",
                     "plein", "plouf", "plupart", "plus", "plusieurs", "plutôt", "possessif", "possessifs", "possible",
                     "possibles", "pouah", "pour", "pourquoi", "pourrais", "pourrait", "pouvait", "prealable",
                     "precisement", "premier", "première", "premièrement", "pres", "probable", "probante", "procedant",
                     "proche", "près", "psitt", "pu", "puis", "puisque", "pur", "pure", "q", "qu", "quand", "quant",
                     "quant-à-soi", "quanta", "quarante", "quatorze", "quatre", "quatre-vingt", "quatrième",
                     "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles", "quelqu'un", "quelque",
                     "quelques", "quels", "qui", "quiconque", "quinze", "quoi", "quoique", "r", "rare", "rarement",
                     "rares", "relative", "relativement", "remarquable", "rend", "rendre", "restant", "reste",
                     "restent", "restrictif", "retour", "revoici", "revoilà", "rien", "s", "sa", "sacrebleu", "sait",
                     "sans", "sapristi", "sauf", "se", "sein", "seize", "selon", "semblable", "semblaient", "semble",
                     "semblent", "sent", "sept", "septième", "sera", "serai", "seraient", "serais", "serait", "seras",
                     "serez", "seriez", "serions", "serons", "seront", "ses", "seul", "seule", "seulement", "si",
                     "sien", "sienne", "siennes", "siens", "sinon", "six", "sixième", "soi", "soi-même", "soient",
                     "sois", "soit", "soixante", "sommes", "son", "sont", "sous", "souvent", "soyez", "soyons",
                     "specifique", "specifiques", "speculatif", "stop", "strictement", "subtiles", "suffisant",
                     "suffisante", "suffit", "suis", "suit", "suivant", "suivante", "suivantes", "suivants", "suivre",
                     "sujet", "superpose", "sur", "surtout", "t", "ta", "tac", "tandis", "tant", "tardive", "te", "tel",
                     "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente", "tes", "tic", "tien",
                     "tienne", "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "touchant", "toujours", "tous",
                     "tout", "toute", "toutefois", "toutes", "treize", "trente", "tres", "trois", "troisième",
                     "troisièmement", "trop", "très", "tsoin", "tsouin", "tu", "té", "u", "un", "une", "unes",
                     "uniformement", "unique", "uniques", "uns", "v", "va", "vais", "valeur", "vas", "vers", "via",
                     "vif", "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voie", "voient", "voilà",
                     "vont", "vos", "votre", "vous", "vous-mêmes", "vu", "vé", "vôtre", "vôtres", "w", "x", "y", "z",
                     "zut", "à", "â", "ça", "ès", "étaient", "étais", "était", "étant", "état", "étiez", "étions",
                     "été", "étée", "étées", "étés", "êtes", "être", "ô", ""]


emojilist = [':)', ':(', ':P', ':p', ':-*', 'XD', '^^', '💀', '👍', '🔥', '💯', '🍆', '🙈', '🙉', '🙊', '😀', '😁', '😂', '🤣', '😃', '😄', '😅', '😆', '😉', '😊', '😋', '😎', '😍', '😘', '😗', '😙', '😚', '☺', '🙂', '🤗', '🤔', '😐', '😑', '😶', '🙄', '😏', '😣', '😥', '😮', '🤐', '😯', '😪', '😫', '😴', '😌', '😛', '😜', '😝', '🤤', '😒', '😓', '😔', '😕', '🙃', '🤑', '😲', '☹', '🙁', '😖', '😞', '😟', '😤', '😢', '😭', '😦', '😧', '😨', '😩', '😬', '😰', '😱', '😳', '😵', '😡', '😠', '😷', '🤒', '🤕', '🤢', '🤧', '😇', '🤠', '🤡', '🤥', '🤓', '😈', '👿', '👹', '👺', '💀', '👻', '👽', '🤖', '💩', '😺', '😸', '😹', '😻', '😼', '😽', '🙀', '😿', '😾', '', '👶', '👧', '👦', '👩', '👨', '👵', '👴', '👲', '👳\u200d', '👱\u200d', '👮\u200d', '👷\u200d', '💂\u200d', '🕵️', '⚕', '🌾', '🍳', '🎓', '🎤', '🏫', '🏭', '💻', '💼', '🔧', '🔬', '🎨', '🚒', '✈', '🚀', '⚖', '👰', '🤵', '👸', '🤴', '🤶', '🎅', '👼', '🤰', '🙇\u200d', '💁\u200d', '🙅\u200d', '🙆\u200d', '🙋\u200d', '🤦\u200d', '🤷\u200d', '🙎\u200d', '🙍\u200d', '💇\u200d', '💆\u200d', '💅', '🤳', '💃', '🕺', '👯\u200d', '🕴', '🚶\u200d', '🏃\u200d', '👫', '👭', '👬', '💑', '❤', '💋\u200d', '👪', '👨\u200d👩\u200d👧', '👨\u200d👩\u200d👧\u200d👦', '👨\u200d👩\u200d👦\u200d👦', '👨\u200d👩\u200d👧\u200d👧', '👩\u200d👩\u200d👦', '👩\u200d👩\u200d👧', '👩\u200d👩\u200d👧\u200d👦', '👩\u200d👩\u200d👦\u200d👦', '👩\u200d👩\u200d👧\u200d👧', '👨\u200d👨\u200d👦', '👨\u200d👨\u200d👧', '👨\u200d👨\u200d👧\u200d👦', '👨\u200d👨\u200d👦\u200d👦', '👨\u200d👨\u200d👧\u200d👧', '👩\u200d👦', '👩\u200d👧', '👩\u200d👧\u200d👦', '👩\u200d👦\u200d👦', '👩\u200d👧\u200d👧', '👨\u200d👦', '👨\u200d👧', '👨\u200d👧\u200d👦', '👨\u200d👦\u200d👦', '👨\u200d👧\u200d👧', '👐', '🙌', '👏', '🤝', '👍', '👎', '👊', '✊', '🤛', '🤜', '🤞', '✌', '🤘', '👌', '👈', '👉', '👆', '👇', '☝', '✋', '🤚', '🖐', '🖖', '👋', '🤙', '💪', '🖕', '✍', '🙏', '💍', '💄', '💋', '👄', '👅', '👂', '👃', '👣', '👁', '👀', '🗣', '👤', '👥']

