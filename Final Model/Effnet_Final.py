import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report

np.random.seed(156)
img_size = (224,224)
batch_size = 16

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../../Dataset_3",
    validation_split= 0.2,
    shuffle= True,
    labels='inferred',
    label_mode='categorical',
    subset='training',
    seed=123,
    image_size= img_size,
    batch_size=batch_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../../Dataset_3",
    validation_split= 0.2,
    shuffle=True,
    subset='validation',
    labels='inferred',
    label_mode='categorical',
    seed=123,
    image_size= img_size,
    batch_size=batch_size
)




import matplotlib.pyplot as plt

class_names = train_ds.class_names
print(class_names)
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")


def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")

# data_rescale = keras.Sequential(
#     [
#         layers.experimental.preprocessing.Rescaling(1.0/255.0)
#     ],
#     name = "rescale_layer"
# )
#
# data_aug = keras.Sequential(
#     [
#      layers.experimental.preprocessing.RandomZoom (height_factor=0.1,width_factor=0.1),
#      layers.experimental.preprocessing.RandomContrast(0.125),
#      layers.experimental.preprocessing.RandomTranslation(height_factor=0.125, width_factor=0.125),
#      layers.experimental.preprocessing.RandomFlip("horizontal"),
#      layers.experimental.preprocessing.RandomRotation(0.15),
#     ],
#     name = "aug_layer"
# )
#
# mainmodel = tf.keras.applications.EfficientNetB0(
#             input_shape= (224,224,3),
#             weights='imagenet',
#             include_top= False,
#             )
# model = tf.keras.Sequential(
#     [
#         keras.Input(shape=(224,224,3)),
#         data_aug,
#         mainmodel,
#         keras.layers.GlobalAveragePooling2D(),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dropout (0.5),
#         keras.layers.Dense(2,activation='softmax'),
#
#     ]
# )
# mainmodel.trainable = False
#
# for layer in model.layers[-20:]:
#             if not isinstance(layer,layers.BatchNormalization):
#                 layer.trainable = True
#
# #hyperparameter : l_rate,batch_size,epoch, optimizer
# optimizer = keras.optimizers.RMSprop(learning_rate=0.1e-4, momentum=0.9)
# model.compile(optimizer= optimizer, loss='categorical_crossentropy',
#                metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
#
# stopper = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience= 8)
# checkpoint = keras.callbacks.ModelCheckpoint(
#     filepath='checkpoint/best_model_EffnetB0_Final',
#     save_best_only= True,
#     monitor='val_accuracy',
#     mode='max'
# )
#
# # model_run = model.fit(train_ds, validation_data=test_ds, epochs=30, callbacks=[checkpoint])
# #
# # plot_hist(model_run)
# # plt.savefig('Accuracy Curve_EffnetB0_Final')
# # plt.show()
# #
# # plot_hist_loss(model_run)
# # plt.savefig('Loss Curve_EffnetB0_Final')
# # plt.show()
# model = tf.keras.models.load_model(filepath='checkpoint/best_model_EffnetB0_Final')
#
# # performance = model.evaluate(test_ds)
# validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "../../Dataset_Validation",
#     shuffle=False,
#     labels='inferred',
#     label_mode='categorical',
#     seed=123,
#     image_size= img_size,
#     batch_size=batch_size
# )
#
# # validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
# #     "../../RL_Dataset",
# #     shuffle=False,
# #     labels='inferred',
# #     label_mode='categorical',
# #     seed=123,
# #     image_size= img_size,
# #     batch_size=batch_size
# # )
#
# y_prediction = model.predict(validation_ds)
#
# predicted_categories = tf.argmax(y_prediction, axis=1)
#
#
# true_categories = tf.concat([y for x, y in validation_ds], axis=0)
# true_categories = tf.argmax(true_categories, axis=1)
#
# confusion = confusion_matrix(predicted_categories, true_categories)
# print(true_categories)
# print(predicted_categories)
# print(confusion)
# target_names = ['O','R']
# print(classification_report(predicted_categories,true_categories, target_names=target_names))
#
# from sklearn.metrics import ConfusionMatrixDisplay
#
# conf_plot =ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=target_names)
# conf_plot.plot()
# plt.show()
# plt.figure(figsize=(10, 10))
# for images, labels in validation_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.title(class_names[predicted_categories[i]])
#         plt.axis("off")
#
# plt.show()
