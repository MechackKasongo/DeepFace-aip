from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Dossier contenant les images d'entraînement (dataset)
root_dir = "dataset"  # Ce dossier doit être présent dans ton projet
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Fonction pour vérifier l'extension de l'image
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fonction pour vérifier si l'image est lisible
def check_image(path):
    img = cv2.imread(path)
    return img is not None

@app.route('/', methods=['GET'])  # autorise uniquement GET ici
def index():
    return jsonify({"message": "DeepFace API is running!"})

@app.route('/predict', methods=['POST'])  # endpoint pour les requêtes POST
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image envoyée'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Fichier vide'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Type de fichier non autorisé. Seuls les fichiers PNG, JPG, JPEG, et GIF sont acceptés.'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(upload_dir, filename)
    
    try:
        file.save(input_path)

        # Vérifie si le fichier est une image valide
        if not check_image(input_path):
            os.remove(input_path)
            return jsonify({'error': 'Fichier invalide ou non lisible comme image'}), 400

        # Parcours du dataset pour identifier la personne
        for person in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person)
            if os.path.isdir(person_dir):
                for image_file in os.listdir(person_dir):
                    reference_path = os.path.join(person_dir, image_file)
                    try:
                        result = DeepFace.verify(input_path, reference_path, model_name='VGG-Face')
                        if result["verified"]:
                            os.remove(input_path)
                            return jsonify({
                                'personne': person,
                                'confiance': f"{100 * (1 - result['distance']):.2f} %"
                            })
                    except Exception as e:
                        print(f"Erreur avec l'image {reference_path} : {e}")
                        continue

        os.remove(input_path)
        return jsonify({'message': "Personne non reconnue"}), 200

    except Exception as e:
        return jsonify({'error': f"Erreur lors du traitement de l'image: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Par défaut 5000
    app.run(host="0.0.0.0", port=port)

