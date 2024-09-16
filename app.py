from flask import Flask, jsonify, request
import nbformat
from nbconvert import PythonExporter
import subprocess

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

    notebook_path = 'D:/Practice_ml/Fifa/Left_Forward.ipynb'  
    return Top_players(notebook_path)
    

@app.route('/Left-Back', methods=['GET'])
def Left_Back():
    notebook_path = 'D:/Practice_ml/Fifa/Left_Back.ipynb'  
    return Top_players(notebook_path)


@app.route('/Left-Mid', methods=['GET'])
def Left_Mid():

    notebook_path = 'D:/Practice_ml/Fifa/Left_Mid.ipynb'  
    return Top_players(notebook_path)

@app.route('/Right-Forward', methods=['GET'])
def Right_Forward():

    notebook_path = 'D:/Practice_ml/Fifa/Right_Forward.ipynb'  
    return Top_players(notebook_path)

@app.route('/Right-Back', methods=['GET'])
def Right_Back():

    notebook_path = 'D:/Practice_ml/Fifa/Right_Back.ipynb'  
    return Top_players(notebook_path)

@app.route('/Right-Mid', methods=['GET'])
def Right_Mid():

    notebook_path = 'D:/Practice_ml/Fifa/Right_Mid.ipynb'  
    return Top_players(notebook_path)

@app.route('/Centre-Back', methods=['GET'])
def Centre_Back():

    notebook_path = 'D:/Practice_ml/Fifa/Centre_Back.ipynb'  
    return Top_players(notebook_path)

# @app.route('/Goal-Keeper', methods=['GET'])
# def Goal_Keeper():

#     notebook_path = 'D:/Practice_ml/Fifa/Goal_Keeper.ipynb'  
#     return Top_players(notebook_path)

if __name__ == '__main__':
    app.run(debug=True)
