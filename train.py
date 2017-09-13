from data_reader import Vocabulary, transform_data, save_vocabularies, Corpus
from model import build_model
from keras.callbacks import ModelCheckpoint







def train(args):

    # ARGUMENTS
    training_file=args.data
    minibatch=64
    max_sent_len_words=30
    epochs=args.epochs
    
    corpus=Corpus(training_file,test_time=False)

    ## VOCABULARY
    vs=Vocabulary()
    vs.build(corpus,min_count_word=args.min_count_word,min_count_sense=args.min_count_sense)
    sentences,x,y=transform_data(corpus,vs,max_sent_len_words)

    model=build_model(vs,max_sent_len_words)
    
    # save model json
    model_json = model.to_json()
    with open(args.model_name+".json", "w") as json_file:
        json_file.write(model_json)
    save_vocabularies(vs,args.model_name+"-vocab.pickle")
    
    # +".{epoch:02d}.h5"
    save_cb=ModelCheckpoint(filepath=args.model_name+".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    
    model.fit(x,y,batch_size=minibatch,epochs=epochs,verbose=1,validation_split=0.1,callbacks=[save_cb])




if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-d', '--data', type=str, required=True, help='Training file')
    g.add_argument('-m', '--model_name', type=str, required=True, help='Name of the saved model')
    g.add_argument('--min_count_word', type=int, default=2, help='Frequency threshold, how many times a word must occur to be included in the vocabulary? (default %(default)d)')
    g.add_argument('--min_count_sense', type=int, default=2, help='Frequency threshold, how many times a verb sense must occur to be included in the vocabulary? (default %(default)d)')
    g.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    args = parser.parse_args()
    
    train(args)
