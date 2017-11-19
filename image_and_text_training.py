from keras.layers import Activation, Dropout, Flatten, Dense, Merge
from keras.models import Sequential

from small_cnn import small_cnn


def make_final_model(model, vec_size, n_classes, do=0.5, l2_strength=1e-5):
    """
    args:
        model : conv model with softmax removed
        vec_size : The size of the vectorized text vector coming from the bag of words 
        n_classes : number of classes
        do : 0.5 Dropout keep probability
        l2_strength :L2 regularization 
    returns:
        final_model : model taking two input sources and making a prediction on n_classes
    """
    
    # top_aux_model takes the vectorized text as input
    top_aux_model = Sequential()
    top_aux_model.add(Dense(vec_size, input_shape=(vec_size,), name='aux_input'))

    # top_model takes output from conv and then adds hidden layers
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:], name='top_flatter'))
    top_model.add(Dense(256, activation='relu', name='top_relu', W_regularizer=l2(l2_strength)))
    top_model.add(Dropout(do))
    top_model.add(Dense(256, activation='sigmoid', name='top_sigmoid', W_regularizer=l2(l2_strength)))
    # this is than added to the conv-model
    model.add(top_model)
    
    #  merge 'model' that creates features from images with 'top_aux_model'
    # that are the bag of words features extracted from the text. 
    merged = Merge([model, top_aux_model], mode='concat')

    # final_model takes the combined feature vectors and add a sofmax classifier to it
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dropout(do))
    final_model.add(Dense(n_classes, activation='softmax'))

    return final_model