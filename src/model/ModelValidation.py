import matplotlib.pyplot as plt


def ModelValidation(history,model,test_set,training_set):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    result = model.evaluate(test_set, steps=1)
    print("Test-set classification accuracy: {0:.2%}".format(result[1]))
    result = model.evaluate(training_set, steps=1)
    print("Train-set classification accuracy: {0:.2%}".format(result[0]))
    
