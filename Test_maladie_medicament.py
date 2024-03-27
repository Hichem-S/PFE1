import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
"""
# Définissez le chemin vers votre dossier dataset
dataset_dir = os.path.abspath("C:\\Users\\hiche\\Desktop\\PFE1\\dataset")
dataset_dirr = os.path.abspath("C:\\Users\\hiche\\Desktop\\PFE1\\dataset")
# Définissez la taille des images et la taille des lots
img_size = (224, 224)
batch_size = 32

# Augmentation des données pour les images
data_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Chargez les données d'entraînement pour les classes "sain" et "malade"
train_data_sain = data_generator.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

train_data_malade = data_generator.flow_from_directory(
    dataset_dirr,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Chargez le modèle de base MobileNetV2 (pré-entraîné sur ImageNet)
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights="imagenet")

# Ajoutez des couches de classification personnalisées
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(len(train_data_malade.class_indices), activation="softmax")(x)

# Créez le modèle
model = Model(inputs=base_model.input, outputs=predictions)

# Compilez le modèle
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entraînez le modèle
model.fit(
    train_data_sain,
    epochs=10,
    validation_data=train_data_malade
)

# Sauvegardez le modèle entraîné
model.save("modele_plantes_maladies.h5")
# Charger le modèle
model = tf.keras.models.load_model("modele_plantes_maladies.h5")"""
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
dataset_dir = os.path.abspath("C:\\Users\\hiche\\Desktop\\PFE1\\dataset")
dataset_dirr = os.path.abspath("C:\\Users\\hiche\\Desktop\\PFE1\\dataset")
# Définissez la taille des images et la taille des lots
img_size = (224, 224)
batch_size = 32
train_data_sain = data_generator.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

train_data_malade = data_generator.flow_from_directory(
    dataset_dirr,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Charger le modèle préalablement entraîné
model = tf.keras.models.load_model("modele_plantes_maladies.h5")

# Chemin vers le dossier contenant les images
dataset_dir = "C:\\Users\\hiche\\Desktop\\PFE1\\dataset"

# Chemin vers l'image de test
image_path = "C:\\Users\\hiche\\Desktop\\PFE1\\Target_Spot.jpg"
# Chargez les données d'entraînement pour les classes "sain" et "malade"
train_data_sain = data_generator.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

train_data_malade = data_generator.flow_from_directory(
    dataset_dirr,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Récupérer les noms des classes
noms_classes = train_data_malade.class_indices.keys()

# Prédire la maladie de la plante
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
classe_predite = np.argmax(predictions[0])
nom_classe_predite = list(noms_classes)[classe_predite]

if nom_classe_predite == "healthy":
    print("L'image est saine.")
else:
    print(f"L'image est malade (maladie : {nom_classe_predite}).")
    # Proposer un médicament
    medicaments = {
       "healthy": "Pas de médicament nécessaire (plante saine)",
        "Bacterial_spot": "Cuivre",
        "Early_blight": "Fongicide à base de cuivre",
        "Late_blight": "Fongicide à base de cuivre",
        "Leaf_Mold": "Fungicide approprié",
        "powdery_mildew": "Fungicide approprié",
        "Septoria_leaf_spot" : "Fongicide à base de cuivre",
        "Spider_mites Two-spotted_spider_mite": "Insecticide approprié",
        "Target_Spot": "Fungicide approprié",
        "Tomato_mosaic_virus": "Herbicide approprié",
        "Tomato_Yellow_Leaf_Curl_Virus": "Herbicide approprié",
    }
    if nom_classe_predite in medicaments:
        medicament_propose = medicaments[nom_classe_predite]
        print(f"Utilisez le médicament : {medicament_propose}")
    else:
        print("Aucun médicament spécifique n'est proposé.")

