def Train(model,training_set,test_set,batch_size,optimizer,epochs,lr):
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history= model.fit(training_set,
                            epochs = epochs,batch_size=batch_size,lr=lr,
                            validation_data = test_set)
    # list all data in history
    print(history.history.keys())
    return history