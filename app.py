from flask import Flask
from routes import main_bp

def create_app():
    """Application factory for Flask"""
    app = Flask(__name__, static_folder='templates', static_url_path='/static')
    
    # Register routes
    app.register_blueprint(main_bp)
    
    return app

app = create_app()

if __name__ == '__main__':
    # Launch the application
    print("Launching World Cup Predictions Web App...")
    app.run(debug=True, port=5000)
