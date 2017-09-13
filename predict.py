from data_reader import Vocabulary, transform_data, load_vocabularies, FORM, Corpus
from model import build_model, load_model
import numpy as np
import sys




def predict(args):

    # ARGUMENTS
    test_file=args.data
    minibatch=64
    max_sent_len_words=30
    
    print("fsdfdfsdfs",file=sys.stderr)

    ## VOCABULARY
    vs=load_vocabularies(args.model_name+"-vocab.pickle")
    
    corpus=Corpus(test_file,test_time=True)
    
    sentences,x,y=transform_data(corpus,vs,max_sent_len_words,test_time=True)
    
    print(len(sentences),flush=True,file=sys.stderr)

    model=load_model(args.model_name)
    predictions=model.predict(x,batch_size=64,verbose=1)
    
    from collections import defaultdict
    # reconstruct file
    last_sentence=None
    for (sent_id,predicate_id),pred in zip(sentences,predictions):
        if last_sentence!=None and sent_id!=last_sentence.id: # new sentence
            corpus.sentences[last_sentence.id]=last_sentence
            last_sentence=corpus.sentences[sent_id]
            last_sentence.arguments=defaultdict(list) # clear arguments to be sure there is nothing
        elif last_sentence==None:
            last_sentence=corpus.sentences[sent_id]
        # add predicted arguments for this predicate
        # predicate_id is the token[ID] of the current predicate
        labels=[vs.idx2label[np.argmax(t)] for t in pred]
        for i,r in enumerate(labels): # i is now token counter, r is predicted role
            if r!="<MASK>" and r!="None":
                last_sentence.arguments[str(i+1)].append((predicate_id,r))

    corpus.save(args.output)     
        



if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-d', '--data', type=str, required=True, help='Training file')
    g.add_argument('-m', '--model_name', type=str, required=True, help='Name of the saved model')
    g.add_argument('-o', '--output', type=str, required=True, help='Output file name')
    
    args = parser.parse_args()
    
    predict(args)
