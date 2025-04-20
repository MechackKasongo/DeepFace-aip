import os
import uuid
import logging
from flask import Flask, request, jsonify
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DATASET_DIR = "dataset"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Vérifie si une extension de fichier est autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Nettoyer les images corrompues
def remove_corrupted_images():
    corrupted = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()
            except:
                os.remove(path)
                logger.warning(f"⚠️ Image corrompue supprimée: {path}")
                corrupted.append(path)
    return corrupted

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Welcome to the face classification API."}), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'ok',
        'message': 'API is running',
    }), 200

@app.route('/cleanup', methods=['POST'])
def cleanup():
    logger.info("🧹 Requête reçue pour le nettoyage du dataset.")
    deleted = remove_corrupted_images()
    return jsonify({
        'status': 'cleanup done',
        'deleted_files': deleted
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("📥 Requête reçue")

    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image reçue'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400

    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(filepath)
        logger.info(f"📸 Image sauvegardée temporairement à {filepath}")

        # Représentations faciales du dataset
        representations = []
        corrupted = []

        for root, dirs, files in os.walk(DATASET_DIR):
            for file_name in files:
                image_path = os.path.join(root, file_name)
                try:
                    rep = DeepFace.represent(img_path=image_path, model_name='VGG-Face', enforce_detection=True)[0]["embedding"]
                    representations.append((image_path, rep))
                except Exception:
                    os.remove(image_path)
                    logger.warning(f"⚠️ Image corrompue supprimée: {image_path}")
                    corrupted.append(image_path)

        # Représentation de l'image reçue
        try:
            input_embedding = DeepFace.represent(img_path=filepath, model_name='VGG-Face', enforce_detection=True)[0]["embedding"]
        except Exception:
            os.remove(filepath)
            return jsonify({"error": "Impossible d'extraire des caractéristiques faciales."}), 500

        # Comparaison avec les images du dataset
        matches = []
        for image_path, rep in representations:
            distance = np.linalg.norm(np.array(rep) - np.array(input_embedding))
            identity = os.path.basename(os.path.dirname(image_path))  # nom du dossier = nom de la personne
            matches.append((identity, distance))

        # Trier par distance
        matches.sort(key=lambda x: x[1])
        top_matches = matches[:3]

        logger.info(f"✅ Prédiction terminée - top 3: {top_matches}")

        return jsonify({
            "status": "ok",
            "top_3": [
                {"identity": identity, "distance": float(distance)}
                for identity, distance in top_matches
            ]
        }), 200

    return jsonify({'error': 'Fichier invalide'}), 400

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    logger.info("==> Lancement du serveur...")
    remove_corrupted_images()  # nettoyage automatique au démarrage
    app.run(host='0.0.0.0', port=10000)
