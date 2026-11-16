from functools import lru_cache
from nltk.corpus import wordnet as wn

ROOTS={wn.synset('object.n.01'),wn.synset('matter.n.03'),wn.synset('thing.n.12'),wn.synset('body_of_water.n.01'),wn.synset('vegetation.n.01'),wn.synset('cloud.n.02'),wn.synset('furniture.n.01')}
SCENE={'image','picture','photo','photograph','scene','view','shot','frame','background','foreground','setting','side','top','bottom',
       'front','back','left','right','center','centre','middle','edge','corner','part','piece','area','region','section','portion','spot',
       'distance','space','place','location','position','surroundings','environment','atmosphere','vicinity','point','end','half',
       'close-up','closeup','detail','element','item','thing','object','individual','one','rest','line','row','set','group','variety',
       'number','lot','couple','pair','bunch','array','collection','series','kind','type','sort','mix','assortment','range','amount',
       'light','lighting','color','colour','feature','ambiance','ambience','presence','layer','shadow','texture','pattern','design','style',
       'tone','hue','shade','glow','reflection','mood','vibe','theme','composition','perspective','angle','sense','contrast','focus','depth',
       'blur','facing','interior','exterior','landscape','surface','structure','material','display','arrangement','decor','decoration',
       'blue','red','green','white','black','yellow','brown','gray','grey','pink','purple','gold','silver','beige','tan','navy','maroon',
       'colors','colours','safety','marking','home'}

FUNCTION=set("""a an the this that these those there here it its it's they them their theirs he him his she her hers we us our you your i me my
mine who whom whose which what where when why how and or but nor so yet for of at by in on to from with without within into onto upon over under
above below between among through during before after about around against along across behind beyond near off out up down as than then also
too very just only even still not no yes all any both each few more most other some such own same can could will would shall should may might
must do does did done doing be is am are was were been being have has had having get gets got make makes made appear appears seem seems look
looks one ones another while if because although though whether either neither like unlike per via etc overall perhaps possibly likely""".split())

def noun_dominant(w):
    key=w.replace(' ','_');n=o=0
    for s in wn.synsets(key):
        c=sum(l.count() for l in s.lemmas() if l.name().lower()==key)
        if s.pos()=='n':n+=c
        else:o+=c
    return n>=o

@lru_cache(maxsize=None)
def lemma(w):
    if w in FUNCTION:return w
    return wn.morphy(w,wn.NOUN) or w

@lru_cache(maxsize=None)
def physical(w):
    if w in SCENE or w in FUNCTION or len(w)<2:return False
    ss=wn.synsets(w.replace(' ','_'),pos=wn.NOUN)[:2]
    return any(r in path for s in ss for path in s.hypernym_paths() for r in ROOTS)
