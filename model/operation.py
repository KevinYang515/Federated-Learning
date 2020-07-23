def init_define(model_x):
    model_x = define_model()

    model_x.compile(optimizer='sgd',
            loss='categorical_crossentropy',
            metrics=['accuracy'])