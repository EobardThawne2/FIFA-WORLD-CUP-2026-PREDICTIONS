# 2026 FIFA World Cup Predictor

A web application that simulates and predicts the outcomes of the 2026 FIFA World Cup. It uses a custom gradient boosting model trained on historical match data and factors in dynamic team statistics like Elo ratings and recent form to forecast group stage standings and knockout brackets.

---

## Features

- **Match Simulations:** Uses a custom-trained Scikit-Learn HistGradientBoosting model to generate win, draw, or loss probabilities based on historical matchups.
- **Dynamic Team Stats:** Tracks teams using dynamic Elo ratings and a 5-match form history.
- **Tournament Simulation:** Simulates the complete 12-group stage phase and the 32-team knockout bracket following the official 2026 format.
- **Fast Performance:** Uses precomputed probabilities and caching so that thousands of tournament runs can be simulated in milliseconds.
- **Interactive UI:** A straightforward plain HTML/CSS/JS frontend to quickly visualize group standings and knockout brackets without heavy client-side frameworks.
- **Vercel Ready:** Pre-configured for deployment on Vercel with optimized memory limits.

## Tech Stack

- **Backend / API:** Python, Flask, Gunicorn
- **Machine Learning:** scikit-learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, Vanilla CSS3, Vanilla JavaScript

## Getting Started

### Prerequisites

You'll need Python 3.8+ installed on your local machine.

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "World Cup"
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app:**
   ```bash
   python app.py
   ```

5. **Open in Browser:**
   Navigate to `http://127.0.0.1:5000` to view the dashboard.

## How the Model Works

The prediction pipeline was trained on a comprehensive historical dataset of international football matches from 1930 to 2022. Rather than using raw data, the model relies on a few key engineered features to represent team strength:

- `home_elo` & `away_elo`: Running Elo ratings updated sequentially after historical matches.
- `home_form_last5` & `away_form_last5`: Weighted point tracking of a team's last 5 games.
- `is_home_advantage`: A binary indicator representing host nation advantage.

If you ever need to retrain the model with fresh data, just run `python train_model.py`. This script will train a new classifier and output a fresh `model.pkl` and `team_stats.pkl`.

## Project Structure

```text
├── app.py                   # Flask server entry point
├── routes.py                # API endpoints and route controllers
├── predictor.py             # Tournament simulation and ML prediction logic
├── train_model.py           # Model training script and feature engineering
├── requirements.txt         # Python dependencies
├── vercel.json              # Vercel deployment configuration
├── schedule_2026.csv        # 2026 World Cup Group Stage match schedule
├── templates/               # Frontend assets
│   ├── fixtures.html
│   ├── bracket.html
│   ├── globals.css
│   └── main.js
└── model.pkl / team_stats.pkl # Serialized model and team state dictionaries
```

## Deploying to Vercel

This repository is ready to be deployed on Vercel out of the box. The `vercel.json` file increases the `maxLambdaSize` to `500mb` and allocates `1024MB` of memory so that the heavier Python machine learning dependencies (`pandas`, `scikit-learn`) can fit perfectly without timing out or crashing.

To deploy, just push your code to a GitHub repository and import the project in your Vercel Dashboard. Vercel will automatically detect the Python Flask setup and handle the rest.

## Acknowledgements

- **Dataset:** The historical match data (`matches_1930_2022.csv`) used to train the prediction model was sourced from Kaggle. Special thanks to the Kaggle community and the creators of the [FIFA Football World Cup dataset](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup) for providing comprehensive international football results.

## License

This project is available under the [MIT License](LICENSE).
