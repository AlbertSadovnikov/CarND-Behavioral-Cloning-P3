import data
import nvidia
from keras.optimizers import Adam


if __name__ == '__main__':
    db = data.database()
    train_db, validation_db = data.validation_split(db)
    model = nvidia.model_x()
    model.summary()
    model.compile(loss='mse', optimizer=Adam(), metrics=['mse', 'accuracy'])
    # uncomment if you want to update trained network
    # model.load_weights('model.h5')
    model.fit_generator(data.generator(train_db, augment=True),
                        validation_data=data.load_data(validation_db),
                        steps_per_epoch=50,
                        epochs=10)
    model.save('model.h5')

