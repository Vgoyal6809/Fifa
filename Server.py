from flask import Flask, jsonify, request
import nbformat
from nbconvert import PythonExporter
import subprocess
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def run_notebook(notebook_path):
    # Load the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Convert the notebook to a Python script
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(nb)

    # Execute the script and collect output
    exec_globals = {}
    exec(script, exec_globals)
    
    # Assuming the notebook has a variable `data` that you want to return
    output_data = exec_globals.get('data', None)
    return output_data

def Top_players(notebook_path):

    # Run the notebook and fetch the data
    data = run_notebook(notebook_path)
    if data.empty:  # This checks if the DataFrame is empty
        return jsonify({"status": "error", "message": "No data found"}), 400
    else:
        return jsonify({"status": "success", "data": data.to_dict(orient='records')})

@app.route('/Left-Forward', methods=['GET'])
def Left_Forward():
    try:
        notebook_path = './Left_Forward.ipynb'
        return Top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/Left-Back', methods=['GET'])
def Left_Back():
    try:
        notebook_path = './Left_Back.ipynb'  
        return Top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/Left-Mid', methods=['GET'])
def Left_Mid():
    try:
        notebook_path = './Left_Mid.ipynb'  
        return Top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/Right-Forward', methods=['GET'])
def Right_Forward():
    try:
        notebook_path = './Right_Forward.ipynb'  
        return Top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/Right-Back', methods=['GET'])
def Right_Back():
    try:
        notebook_path = './Right_Back.ipynb'  
        return Top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/Right-Mid', methods=['GET'])
def Right_Mid():
    try:
        notebook_path = './Right_Mid.ipynb'  
        return Top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/Centre-Back', methods=['GET'])
def Centre_Back():
    try:
        notebook_path = './Centre_Back.ipynb'  
        return Top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/Stricker', methods=['GET'])
def Stricker():
    try:
        notebook_path = './Stricker.ipynb'  
        return Top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/Goal-Keeper', methods=['GET'])
def Goal_Keeper():
    try:
        notebook_path = './Goal_Keeper.ipynb'  
        return Top_players(notebook_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)))

