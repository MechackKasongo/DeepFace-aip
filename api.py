from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import uuid
from PIL import Image
import numpy as np

app = Flask(__name__)

# Dossier pour stocker temporairement les images reçues
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return 'API DeepFace en ligne et opérationnelle !'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Requête reçue")

        # Vérifie qu'une image a été envoyée
        if 'image' not in request.files:
            print("⚠Aucune image reçue")
            return jsonify({'error': 'Aucune image reçue'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            print("Nom de fichier vide")
            return jsonify({'error': 'Nom de fichier vide'}), 400

        # Sauvegarde temporairement l'image
        image_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.jpg")
        image_file.save(input_path)
        print(f"Image sauvegardée temporairement à {input_path}")

        # Répertoire contenant les visages de référence
        dataset_path = "dataset"

        # Lancer la reconnaissance faciale
        result = DeepFace.find(img_path=input_path, db_path=dataset_path, model_name='VGG-Face')

        if result and isinstance(result, list) and len(result[0]) > 0:
            # On récupère la correspondance avec le plus haut score (plus petite distance)
            top_match = result[0].iloc[0]
            identity = os.path.basename(os.path.dirname(top_match["identity"]))
            print(f"Correspondance trouvée : {identity}")
            response = {'identity': identity}
        else:
            print("Aucun visage correspondant trouvé.")
            response = {'identity': 'Inconnu'}

        # Nettoyage du fichier temporaire
        if os.path.exists(input_path):
            os.remove(input_path)

        return jsonify(response)

    except Exception as e:
        print(f"Erreur : {str(e)}")
        return jsonify({'error': f"Erreur lors du traitement de l'image: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)




# from flask import Flask, request, jsonify
# from deepface import DeepFace
# import os
# import cv2
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Dossier contenant les images d'entraînement (dataset)
# root_dir = "dataset"  # Ce dossier doit être présent dans ton projet
# upload_dir = "uploads"
# os.makedirs(upload_dir, exist_ok=True)

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # Fonction pour vérifier l'extension de l'image
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Fonction pour vérifier si l'image est lisible
# def check_image(path):
#     img = cv2.imread(path)
#     return img is not None

# @app.route('/', methods=['GET'])  # autorise uniquement GET ici
# def index():
#     return jsonify({"message": "DeepFace API is running!"})

# @app.route('/predict', methods=['POST'])  # endpoint pour les requêtes POST
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'Aucune image envoyée'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'Fichier vide'}), 400

#     if not allowed_file(file.filename):
#         return jsonify({'error': 'Type de fichier non autorisé. Seuls les fichiers PNG, JPG, JPEG, et GIF sont acceptés.'}), 400

#     filename = secure_filename(file.filename)
#     input_path = os.path.join(upload_dir, filename)
    
#     try:
#         file.save(input_path)

#         # Vérifie si le fichier est une image valide
#         if not check_image(input_path):
#             os.remove(input_path)
#             return jsonify({'error': 'Fichier invalide ou non lisible comme image'}), 400

#         # Parcours du dataset pour identifier la personne
#         for person in os.listdir(root_dir):
#             person_dir = os.path.join(root_dir, person)
#             if os.path.isdir(person_dir):
#                 for image_file in os.listdir(person_dir):
#                     reference_path = os.path.join(person_dir, image_file)
#                     try:
#                         result = DeepFace.verify(input_path, reference_path, model_name='VGG-Face')
#                         if result["verified"]:
#                             os.remove(input_path)
#                             return jsonify({
#                                 'personne': person,
#                                 'confiance': f"{100 * (1 - result['distance']):.2f} %"
#                             })
#                     except Exception as e:
#                         print(f"Erreur avec l'image {reference_path} : {e}")
#                         continue

#         os.remove(input_path)
#         return jsonify({'message': "Personne non reconnue"}), 200

#     except Exception as e:
#         return jsonify({'error': f"Erreur lors du traitement de l'image: {str(e)}"}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))  # Par défaut 5000
#     app.run(host="0.0.0.0", port=port)

