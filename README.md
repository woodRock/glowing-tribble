# RosterForge

RosterForge is a web-based application for generating and managing staff rosters. It features an automatic roster generation system based on the stable marriage algorithm and a visual manual editor inspired by the Overwatch hero select screen.

## How to Run the Application

This project is a monorepo containing a `server` (backend) and a `client` (frontend). You will need to run them in two separate terminal windows.

### Running the Backend Server

1.  **Navigate to the server directory:**
    ```bash
    cd roster-forge/server
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Start the server:**
    ```bash
    npm start
    ```
    The backend server will be running on `http://localhost:4000`.

### Running the Frontend Client

1.  **Navigate to the client directory:**
    ```bash
    cd roster-forge/client
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Start the client:**
    ```bash
    npm run dev
    ```
    The frontend application will be running on `http://localhost:5173` (or the next available port).
# glowing-tribble
