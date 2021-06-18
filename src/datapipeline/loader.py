from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self):
        pass

    def init_generator(self):
        # Generate batches of images with augmentation for training set
        # Data-augmentation only applies to train set
        self.aug = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
        )
        self.no_aug = ImageDataGenerator()
    
    def load_data(self):
        BATCH_SIZE = 512

        img_size = (48, 48) # this is a requirement of MobileNet
        train_data = self.aug.flow_from_directory(
            directory='data/train',
            color_mode='rgb',
            target_size=img_size,
            batch_size=BATCH_SIZE, 
            shuffle=True,
            seed=18,
            class_mode='categorical'
        )
        val_data = self.no_aug.flow_from_directory(
            directory='data/val',
            color_mode='rgb',
            target_size=img_size,
            batch_size=BATCH_SIZE, 
            shuffle=True,
            seed=18,
            class_mode='categorical'
        )
        test_data = self.no_aug.flow_from_directory(
            directory='data/test',
            color_mode='rgb',
            target_size=img_size,
            batch_size=BATCH_SIZE, 
            shuffle=True,
            seed=18,
            class_mode='categorical'
        ) 
        return train_data, val_data, test_data