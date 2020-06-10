from keras.preprocessing.image_dev import ImageDataGenerator

train_path = "training/0"
save_path = "traininf/AUG_0"

train_datagen = ImageDataGenerator(
#	rescale=1./ 255,
#	rotation_range=30,
#	width_shift_range=0.2,
#	height_shift_range=0.2,
#	shear_range=0.2,
#	zoom_range=0.2,
	fill_mode = 'nearest',
	cval = 0,
	horizontal_flip=True,
	vertical_flip=True,
#	blurring=True,
#	brightness=0.4,
#	saturation=0.2
)

i = 0
for batch in train_datagen.flow_from_directory(train_path, batch_size = 1600, target_size=(600,800), save_to_dir=save_path):
    i += 1   
    if i > 1:
	#if i > 1: 2x Aug
	#if i > 4: 5x Aug
	#if i > 0: 1x Aug
        break
