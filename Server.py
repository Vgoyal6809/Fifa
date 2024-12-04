from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import json
import nbformat
from nbconvert import PythonExporter
import pickle

from Second_Use_Case import analyze_video  # Import your analysis logic

app = Flask(__name__)
CORS(app)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = './static/uploaded_videos'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize global player data
player_data = {}

# Notebook-related functions
def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(nb)

    exec_globals = {}
    exec(script, exec_globals)

    # Fetch 'data' from notebook execution
    return exec_globals.get('data', None)

def save_data_pickle(data, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

def load_data_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def get_top_players(notebook_path):
    pickle_path = notebook_path.replace('.ipynb', '.pkl')

    if os.path.exists(pickle_path):
        data = load_data_pickle(pickle_path)
    else:
        data = run_notebook(notebook_path)
        if data is not None:
            save_data_pickle(data, pickle_path)

    if data.empty:  # Check if DataFrame is empty
        return jsonify({"status": "error", "message": "No data found"}), 400
    else:
        return jsonify({"status": "success", "data": data.to_dict(orient='records')})

NOTEBOOK_PATHS = {
    "Left-Forward": './Left_Forward.ipynb',
    "Left-Back": './Left_Back.ipynb',
    "Left-Mid": './Left_Mid.ipynb',
    "Right-Forward": './Right_Forward.ipynb',
    "Right-Back": './Right_Back.ipynb',
    "Right-Mid": './Right_Mid.ipynb',
    "Centre-Back": './Centre_Back.ipynb',
    "Striker": './Striker.ipynb',
    "Goal-Keeper": './Goal_Keeper.ipynb',
}

# Flask Routes

@app.route('/upload', methods=['POST'])
def upload_video():
    print(request.files)
    global player_data
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(video_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(file_path)

    analyze_video(file_path, player_data)

    return(player_data)

@app.route('/<position>', methods=['GET'])
def get_position_data(position):
    try:
        notebook_path = NOTEBOOK_PATHS.get(position)
        if not notebook_path:
            return jsonify({"status": "error", "message": f"Position '{position}' not found"}), 404
        return get_top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)), debug=True)
