import pandas as pd
import numpy as np
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, \
Dropout, Activation, Input, Flatten, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import gensim

def cnn_model(FILTER_SIZES, \
              # filter sizes as a list
              MAX_NB_WORDS, \
              # total number of words
              MAX_DOC_LEN, \
              # max words in a doc
              EMBEDDING_DIM=200, \
              # word vector dimension
              NUM_FILTERS=64, \
              # number of filters for all size
              DROP_OUT=0.5, \
              # dropout rate
              NUM_OUTPUT_UNITS=1, \
              # number of output units
              NUM_DENSE_UNITS=100,\
              # number of units in dense layer
              PRETRAINED_WORD_VECTOR=None,\
              # Whether to use pretrained word vectors
              LAM=0.0):            
              # regularization coefficient
    
    main_input = Input(shape=(MAX_DOC_LEN,), \
                       dtype='int32', name='main_input')
    
    if PRETRAINED_WORD_VECTOR is not None:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        # use pretrained word vectors
                        weights=[PRETRAINED_WORD_VECTOR],\
                        # word vectors can be further tuned
                        # set it to False if use static word vectors
                        trainable=True,\
                        name='embedding')(main_input)
    else:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        name='embedding')(main_input)
    # add convolution-pooling-flat block
    conv_blocks = []
    for f in FILTER_SIZES:
        conv = Conv1D(filters=NUM_FILTERS, kernel_size=f, \
                      activation='relu', name='conv_'+str(f))(embed_1)
        conv = MaxPooling1D(MAX_DOC_LEN-f+1, name='max_'+str(f))(conv)
        conv = Flatten(name='flat_'+str(f))(conv)
        conv_blocks.append(conv)
    
    if len(conv_blocks)>1:
        z=Concatenate(name='concate')(conv_blocks)
    else:
        z=conv_blocks[0]
        
    drop=Dropout(rate=DROP_OUT, name='dropout')(z)

    dense = Dense(NUM_DENSE_UNITS, activation='relu',\
                    kernel_regularizer=l2(LAM),name='dense')(drop)
    preds = Dense(NUM_OUTPUT_UNITS, activation='sigmoid', name='output')(dense)
    model = Model(inputs=main_input, outputs=preds)
    
    model.compile(loss="binary_crossentropy", \
              optimizer="adam", metrics=["accuracy"]) 
    
    return model

# Q1
def sentiment_cnn(file_path):
    
    data=pd.read_csv(file_path, header=0)

    MAX_NB_WORDS=446
    MAX_DOC_LEN=179
    EMBEDDING_DIM=300

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(data["text"])
    sequences = tokenizer.texts_to_sequences(data["text"])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_DOC_LEN, padding='post', truncating='post')
    
    mlb = MultiLabelBinarizer()
    Y=mlb.fit_transform(data.label.astype(str).tolist())
    output_units_num=len(mlb.classes_)
    
    FILTER_SIZES=[2,3,4]
    BTACH_SIZE = 64
    NUM_EPOCHES = 20
    num_filters=64
    dense_units_num= num_filters*len(FILTER_SIZES)
    
    BEST_MODEL_FILEPATH="best_model"

    X_train, X_test, Y_train, Y_test = train_test_split(\
                    padded_sequences, Y, test_size=0.2, random_state=0)

    model=cnn_model(FILTER_SIZES, MAX_NB_WORDS, MAX_DOC_LEN, EMBEDDING_DIM=EMBEDDING_DIM, NUM_FILTERS=num_filters,
                    NUM_OUTPUT_UNITS=output_units_num, NUM_DENSE_UNITS=dense_units_num)

    earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
    checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_loss', \
                                 verbose=2, save_best_only=True, mode='min')

    training=model.fit(X_train, Y_train, \
              batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, \
              callbacks=[earlyStopping, checkpoint],\
              validation_data=[X_test, Y_test], verbose=2)
    
    # load the best model and predict
    model.load_weights("best_model")
    pred=model.predict(X_test)
    pred=np.where(pred>0.5, 1, 0)
    # Accuracy
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Generate performance report
    print(classification_report(Y_test, pred, target_names=mlb.classes_))
    
    return

# Q2
def improved_sentiment_cnn(file_path):
    
    data=pd.read_csv(file_path, header=0)
    
    # load google pretrained word vector
    kv = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    MAX_NB_WORDS=446
    MAX_DOC_LEN=179
    EMBEDDING_DIM=300

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(data["text"])

    # looking for words from pretraind word vector, save them in embedding_matrix
    embedding_matrix = np.zeros((MAX_NB_WORDS+1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        # if word_index is above the max number of words, ignore it
        if i >= MAX_NB_WORDS:
            continue
        if word in kv.wv:
            embedding_matrix[i]=kv.wv[word]

    sequences = tokenizer.texts_to_sequences(data["text"])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_DOC_LEN, padding='post', truncating='post')
    
    mlb = MultiLabelBinarizer()
    Y=mlb.fit_transform(data.label.astype(str).tolist())
    output_units_num=len(mlb.classes_)
    
    FILTER_SIZES=[2,3,4]
    BTACH_SIZE = 64
    NUM_EPOCHES = 20
    num_filters=64
    dense_units_num= num_filters*len(FILTER_SIZES)
    
    BEST_MODEL_FILEPATH="best_model"

    X_train, X_test, Y_train, Y_test = train_test_split(padded_sequences, Y, test_size=0.2, random_state=0)

    model=cnn_model(FILTER_SIZES, MAX_NB_WORDS, MAX_DOC_LEN, EMBEDDING_DIM=EMBEDDING_DIM, NUM_FILTERS=num_filters,
                    NUM_OUTPUT_UNITS=output_units_num, NUM_DENSE_UNITS=dense_units_num)

    earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
    checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_loss', 
                                 verbose=2, save_best_only=True, mode='min')

    training=model.fit(X_train, Y_train, batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, 
                       callbacks=[earlyStopping, checkpoint],validation_data=[X_test, Y_test], verbose=2)
    
    # load the best model and predict
    model.load_weights("best_model")
    pred=model.predict(X_test)
    pred=np.where(pred>0.5, 1, 0)
    # Accuracy
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Generate performance report
    print(classification_report(Y_test, pred, target_names=mlb.classes_))
    
    return

# main
if __name__ == "__main__":  
    
    file_path = 'amazon_review_500.csv'
    sentiment_cnn(file_path)
    improved_sentiment_cnn(file_path)
