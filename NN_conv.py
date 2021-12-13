from tensorflow import keras
from sklearn.utils import shuffle

def NN_conv_model(layers, optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']):
    
    '''
    This function receives the 'layers', 'optimizer', 'loss' y 'metrics' which are necessaries to create a NN or a CNN.
    Parameters:
    - layers: of the NN or CNN
    - optimizer: optimizer of the model. 'adam' by default
    - loss: loss function. 'sparse_categorical_crossentropy' by default.
    - metrics: metrics for the evaluation. It is an iterable. ['accuracy'] by default.

    Example of layers for a CNN:
    
    layers = [keras.layers.Conv2D(first_layer_conv, (3,3), activation=activation, input_shape=image_size),
            keras.layers.MaxPooling2D(pool_size=(2,2)),

            keras.layers.Conv2D(second_layer_conv, (3,3), activation=activation),
            keras.layers.MaxPooling2D(pool_size=(2,2)),

            keras.layers.Flatten(),
            keras.layers.Dense(first_layer_NN, activation=activation),
            keras.layers.Dense(second_layer_NN, activation=activation),
            keras.layers.Dense(len(class_names_label), activation=final_activation)
        ]
    '''

    model = keras.Sequential(layers)

    model.compile(optimizer = optimizer,
                loss = loss,
                metrics = metrics
                )
    
    return model


def model_train(model, X, y, batch_size= 64, epochs= 15, validation_split= 0.1, shuffle_ = False, conv = False):
    '''
    This function receives a model and trains it. Then, it returns the history of the trained model.
    Variables:
    - model: prediction model. It must be a NN or a CNN.
    - X: training values. It is an iterable.
    - y: training labels. It is an iterable.
    - batch_size: batch_size parameter of the NN. 64 by default.
    - epoch: epochs for the training model. 15 by default.
    - validation_split: portion for the validation. 0.1 by default.
    - shuffle_: True for making a 'shuffle' of the values. False by default.
    - conv: False if it is a NN and True if it is a CNN (images). False by default.
    '''
    if shuffle_ == True:
        X, y = shuffle(X, y, random_state=42)
    
    if conv == True:
        X = X/255

    history = model.fit(X,
                    y,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_split = validation_split
                    )
    return history


def load_model(path):
    
    '''
    The function loads a keras model of the path provided.
    Parameters:
    - path: path of the model.
    '''

    model = keras.models.load_model(path)

    return model


def save_model(model, path, model_name):

    '''
    The function saved the keras model received in the path provided.
    Parameters:
    - model: the trained model
    - path: saving path
    - model_name: name of the model
    '''

    model.save(path + '\\' + model_name + '.h5')