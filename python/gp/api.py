from flask import Flask, request, jsonify
from gp_rostering import run_gp_rostering
import os

app = Flask(__name__)

# Manual CORS setup to avoid dependency issues
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/generate_roster', methods=['POST'])
def generate_roster():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    staff_data = data.get('staff_data')
    shift_data = data.get('shift_data')
    generations = data.get('generations', 10)
    population_size = data.get('population_size', 50)
    cxpb = data.get('cxpb', 0.5)
    mutpb = data.get('mutpb', 0.2)

    if not staff_data or not shift_data:
        return jsonify({"error": "Missing 'staff_data' or 'shift_data' in request"}), 400

    try:
        # Run the GP algorithm to find the best heuristic and generate a roster
        best_roster_details, best_heuristic_tree = run_gp_rostering(
            staff_data, shift_data, requests=data.get('requests'), generations=generations, population_size=population_size, cxpb=cxpb, mutpb=mutpb
        )
        return jsonify({
            "status": "success",
            "roster_details": best_roster_details,
            "best_heuristic_tree": best_heuristic_tree
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "Roster Forge GP API is running!"

if __name__ == '__main__':
    # For development, run with debug=True
    # In production, use a production-ready WSGI server like Gunicorn
    app.run(debug=True, port=5000)

