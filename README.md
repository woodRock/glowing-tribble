# RosterForge

RosterForge is a web-based application for generating and managing staff rosters. It features an automatic roster generation system based on various evolutionary computation algorithms and a visual manual editor.

## How to Run the Application

This project now uses a Python Flask server (for algorithmic computations and data serving) and a React frontend client. You will need to run them in two separate terminal windows.

### 1. Run the Python Backend Server

This server provides the evolutionary computation algorithms and serves benchmark problem data.

1.  **Ensure Python dependencies are installed:**
    Make sure you have `flask` and `flask-cors` installed. If not, run:
    ```bash
    pip install flask flask-cors
    ```
    (Note: If you have multiple Python versions, use `python3 -m pip install flask flask-cors` to ensure it's installed for the correct interpreter.)

2.  **Navigate to the project root directory:**
    ```bash
    cd /Users/woodj/Desktop/roster-forge
    ```

3.  **Start the Python server:**
    ```bash
    python3 server/universal_api.py
    ```
    The Python server will be running on `http://localhost:5001`. Check your terminal for output confirming it has started.

### 2. Run the Frontend Client

1.  **Navigate to the client directory:**
    ```bash
    cd client
    ```

2.  **Install JavaScript dependencies:**
    ```bash
    npm install
    ```

3.  **Start the client:**
    ```bash
    npm run dev
    ```
    The frontend application will typically be running on `http://localhost:5173` (or the next available port). Open this URL in your web browser.

Once both are running, you can select a benchmark problem and an algorithm in the client application to generate rosters.