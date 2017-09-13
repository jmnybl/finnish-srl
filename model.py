from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Reshape, Flatten, concatenate, Bidirectional, TimeDistributed, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras import optimizers
import keras.backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.callbacks import Callback

from collections import Counter
from math import ceil
import sys
import gzip
import numpy as np
import random
import lwvlib
import json

# parameters
word_embedding_size=200
recurrent_inner=512
stack=1

def read_pretrained_vectors(vocabulary,vector_model,max_words=1000000):
    print("Loading pretrained word embeddings from "+vector_model)
    model=lwvlib.load(vector_model,max_words,max_words)
    vectors=np.zeros((len(vocabulary),model.vectors.shape[1]),np.float)
    for word,idx in vocabulary.items():
        if idx==0 or idx==1 or idx==2:
            continue # mask and unk and end
        midx=model.get(word)
        if midx is not None:
            vectors[idx]=model.vectors[midx] # TODO: normalize?
            
    return vectors
    
# Implements a layer which picks the predicate word from the lstm sequence
# Input shape: (minibatch,1), (minibatch, max_sequence, lstm_width)
# Output shape: (minibatch, 1, lstm_width)
#def vector_from_sequence(inputs):
#    p,lstm=inputs[0],inputs[1]
#    v=lstm[:,1,:]
#    print(K.shape(v)[1])
#    return Flatten()(v)

#is=Lambda(vector_from_sequence,vector_from_sequence_dim)([i,lstms[-1]])
#concat_lstm_sense=concatenate([lstms[-1],is],axis=-1) 
    
#def vector_from_sequence_dim(lstm_input_shape):
#    return (lstm_input_shape[1][0],30,lstm_input_shape[1][2])



def build_model(vs,max_sent_len_words):

    print("Building model",file=sys.stderr)

    # inputs
    input_words=Input(shape=(max_sent_len_words,),name="input_words")
    input_predicate=Input(shape=(max_sent_len_words,),name="predicate_vector") # vector where predicate word is 1. other words are zero
    input_predicate_sense=Input(shape=(max_sent_len_words,),name="predicate_sense") # vector where predicate word is 1. other words are zero
    
    # embeddings
    pretrained=read_pretrained_vectors(vs.words,"/home/jmnybl/word_embeddings/pb34_wf_200_v2.bin") # 
    
    word_embeddings=Embedding(vs.vocab_size, word_embedding_size, name="word_embeddings", mask_zero=False, weights=[pretrained])(input_words)
    binary=Embedding(3, 1, name="binary_embeddings", mask_zero=False, weights=[np.array([0.0,0.0,1.0]).reshape((3,1))], trainable=False)(input_predicate)
    
    sense_embeddings=Embedding(len(vs.predicate_senses), word_embedding_size, name="sense_embeddings", mask_zero=False)(input_predicate_sense)
    
    
    # concatenate word embeddings and predicate vector
    concat_embedding=concatenate([word_embeddings,binary],axis=-1)
    
    lstms=[concat_embedding]
    
    # recurrent
    for _ in range(stack):
    
        if len(lstms)>1:
            lstm_in=concatenate(lstms,axis=-1)
        else:
            lstm_in=lstms[0]
        
        lstm_out=Bidirectional(LSTM(recurrent_inner,name="bilstm",return_sequences=True, activation="relu"), merge_mode="ave")(lstm_in)
        lstms.append(lstm_out)
         
        
    concat_lstm_sense=concatenate([lstms[-1],sense_embeddings],axis=-1)
    
    
    # softmax 
    predictions=TimeDistributed(Dense(len(vs.argument_roles),activation="softmax",name="prediction_layer"))(concat_lstm_sense)
    
    model=Model(inputs=[input_words,input_predicate,input_predicate_sense], outputs=[predictions])
    
#    adam=optimizers.Adam(beta_2=0.9)
    model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
    
    print(model.summary())
    from keras.utils import plot_model

    plot_model(model,to_file="model.png",show_shapes=True)
    
    return model
    
    
    
def load_model(model_name):

    with open(model_name+".json", "rt", encoding="utf-8") as f:
        model=model_from_json(f.read())
    model.load_weights(model_name+".h5")

    return model
    
    
    
    
    
    
    
