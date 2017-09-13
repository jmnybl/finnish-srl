from collections import defaultdict, Counter
import pickle
ID,FORM,LEMMA,UPOS,POS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)
SID,SFORM,SLEMMA,SPLEMMA,SPOS,SPPOS,SFEAT,SPFEAT,SHEAD,SPHEAD,SDEPREL,SPDEPREL,PREDICATE,SENSE=range(14)
import sys

class Vocabulary(object):

    def __init__(self):
        pass
        
        
    def word_idx(self,word):
        return self.words.get(word,self.words["<UNK>"])
                
               
        
    def build(self,corpus,min_count_word,min_count_sense):
        print("Building vocabulary",file=sys.stderr)
        words=Counter()
        predicate_senses=Counter()
        self.argument_roles={"<MASK>":0,"None":1}
        tmp_tokens=[]
        total_words=0
        for sent_id,sentence in corpus.iterate():
            for token in sentence.tokens:
                tmp_tokens.append(token[FORM])
                total_words+=1
            if len(tmp_tokens)>10000:
                words.update(tmp_tokens)
            for (idx,sense) in sentence.predicates:
                predicate_senses.update([sense]) #auto-grows
                for (idx,arg) in sentence.arguments[idx]:
                    self.argument_roles.setdefault(arg,len(self.argument_roles)) #auto-grows
        if tmp_tokens:
            words.update(tmp_tokens)
        self.words={"<MASK>":0,"<END>":1,"<UNK>":2}
        for w,count in words.most_common():
            if count<min_count_word:
                break
            self.words[w]=len(self.words)
        self.predicate_senses={"<MASK>":0,"<UNK>":1}
        for s,count in predicate_senses.most_common():
            if count<min_count_sense:
                break
            self.predicate_senses[s]=len(self.predicate_senses)
        self.vocab_size=len(self.words)
        self.idx2label={}
        for key,value in self.argument_roles.items():
            self.idx2label[value]=key
        print(total_words,"words in the training corpus.")
        print("Created vocabulary with {x} words, {y} predicate senses, and {z} different roles.".format(x=self.vocab_size,y=len(self.predicate_senses),z=len(self.argument_roles)))
        
        
def save_vocabularies(vs, fname):
    with open(fname,"wb") as f:
        pickle.dump(vs,f,pickle.HIGHEST_PROTOCOL)
            
def load_vocabularies(fname):
    with open(fname,"rb") as f:
        return pickle.load(f)
        
class Corpus(object):

    def __init__(self,fname,test_time=False):
        self.sentences={}
        self.sentence_ids=[]
        with open(fname,"rt",encoding="utf-8") as f:
            for sent in self.read_corpus(f,test_time):
                self.sentences[sent.id]=sent
                self.sentence_ids.append(sent.id)
        print("Corpus has {x} sentences.".format(x=len(self.sentence_ids)))
                
                
    def iterate(self):
        for idx in self.sentence_ids:
            yield idx, self.sentences[idx]
            
    
    def read_corpus(self,f,test_time):
        comments=[]
        tokens=[]
        predicates=[] # (predicate_id,sense)
        arguments=defaultdict(list) # key: predicate_id, value: (argument_id,role)
        sentence_counter=0
        for line in f:
            line=line.strip()
            if not line: # new sentence
                if tokens:
                    yield Sentence(sentence_counter,tokens,predicates,arguments,comments=comments)
                    sentence_counter+=1
                    comments,tokens,predicates,arguments=[],[],[],defaultdict(list)
            elif line.startswith("#"):
                comments.append(line)
            else: #normal line
                cols=line.split("\t")
                tokens.append(cols)
                # senses
                if "PBSENSE" in cols[MISC]:
                    for m in cols[MISC].split("|"):
                        if m.startswith("PBSENSE"):
                            sense=m.split("=",1)[-1]
                            predicates.append((cols[ID],sense)) # ID (str) based indexing, not list based
                if test_time==True: # do not read arguments
                    continue
                # arguments
                if "PBArg" in cols[DEPS]:
                    for a in cols[DEPS].split("|"):
                        if "PBArg" in a:
                            idx,arg=a.split(":",1) # idx is predicate id
                            arguments[idx].append((cols[ID],arg))
        else:
            if tokens:
                yield Sentence(sentence_counter,tokens,predicates,arguments,comments=comments)
                
                
    def save(self,fname):

        with open(fname,"wt",encoding="utf-8") as f:
            for sent_id, sent in self.iterate():
                sent.print_conllu(f)               
               
        
class Sentence(object):

    def __init__(self,idx,tokens,predicates,arguments=defaultdict(list),comments=[]):
        """ predicates = list of (predicate_id,sense) -tuples """
        self.id=idx
        self.tokens=tokens
        self.comments=comments
        self.predicates=predicates
        self.arguments=arguments
        
        
    def print_conllu(self,outfile):
        #print(self.tokens,self.predicates,self.arguments)
        for cols in self.tokens:
        
            # mask sense for now TODO
            
            # remove gold standard arguments
            cols[DEPS]="_"
            
            if cols[ID] in self.predicates:
                cols[MISC]="PBSENSE=empty.1"
            args=[]
            if cols[ID] in self.arguments:
                for (pidx,role) in self.arguments[cols[ID]]:
                    args.append(pidx+":"+role)
            if len(args)==0:
                cols[DEPS]="_"
            else:
                cols[DEPS]="|".join(args)
            print("\t".join(cols),flush=True,file=outfile)
        print("",flush=True,file=outfile)
        
        
        
        
    def print_sdp(self,outfile):

        predicate_counter={}
        for key, val in sorted(self.predicates.items(),key=lambda x: int(x[0])):
            predicate_counter[key]=len(predicate_counter)
        
        for i,token in enumerate(self.tokens):
            print_line(self,i,token,predicate_counter)
        print("")
            
            
    def print_line_sdp(self,i,cols,predicate_counter):
        new_cols=["_"]*(14+len(self.predicates))
        for c,u in zip([ID,FORM,LEMMA,UPOS,FEAT,HEAD,DEPREL], [SID,SFORM,SLEMMA,SPOS,SFEAT,SHEAD,SDEPREL]):
            new_cols[u]=cols[c]
        if cols[ID] in self.predicates:
            new_cols[PREDICATE]="Y"
            new_cols[SENSE]=self.predicates[cols[ID]]
        if cols[ID] in self.arguments:
            for pidx,role in self.arguments[cols[ID]]:
                p_count=predicate_counter[pidx]
                new_cols[14+p_count]=role
        print("\t".join(new_cols)) 
            
        
        
        
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
  
def transform_data(corpus,vs,max_sent_len_word,test_time=False):

    sentences=[] # (sentence_id,predicate_id)
    input_sequences=[] # token sequence for each example, after last token special <END> token is included
    output_sequences=[] # target sequence for each example <MASK> label for out of sentence labels
    predicate_vectors=[] # one hot vector, mark which of the words is the current predicate
    predicate_senses=[] # for each example, predicate sense is repeated for each word in the sentence
    ppp=[]
    
    for sent_idx,sent in corpus.iterate():
        
        for (pidx,sense) in sent.predicates:
            # this is now one example
            if int(pidx) > max_sent_len_word: # predicate is out of max sentence len, skip if training
                print("skipping sentence",flush=True,file=sys.stderr)
                continue
            
                
            sentences.append((sent_idx,pidx)) # list of sent_ids and predicate_ids so that we can map predictions
            
            # initialize everything with zeros/empty
            words=[]
            senses=[]
            roles=np.zeros((max_sent_len_word,len(vs.argument_roles))) # categorical representation
            # binary predicate vector
            p_vector=[0]*max_sent_len_word
            p_vector[int(pidx)-1]=2
            ppp.append(int(pidx)-1)
            
            # collect all words that are arguments for this pedicate
            proles={}
            for aidx,r in sent.arguments[pidx]:
                proles[aidx]=r
                
            for i in range(max_sent_len_word):
                if i>=len(sent.tokens):
                    # out of sentence, role is <MASK>
                    roles[i,vs.argument_roles["<MASK>"]]=1.0
                    continue
                if i!=int(pidx)-1:
                    p_vector[i]=1
                token=sent.tokens[i]
                words.append(vs.word_idx(token[FORM]))
                r=vs.argument_roles[proles.get(token[ID],"None")]
                roles[i,r]=1.0
                senses.append(vs.predicate_senses.get(sense,vs.predicate_senses["<UNK>"]))
                
            if len(sent.tokens) < max_sent_len_word:
                words.append(vs.word_idx("<END>"))
            # add to the batch
            input_sequences.append(words)
            output_sequences.append(roles)
            predicate_senses.append(senses)
            predicate_vectors.append(p_vector)
    
    return sentences,{"input_words":pad_sequences(input_sequences,maxlen=max_sent_len_word,padding="post",truncating="post"),"predicate_sense":pad_sequences(predicate_senses,maxlen=max_sent_len_word,padding="post",truncating="post"),"predicate_vector":np.array(predicate_vectors),"ppp":ppp},np.array(output_sequences)
            









