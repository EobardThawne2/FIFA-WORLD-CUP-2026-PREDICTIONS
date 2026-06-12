import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import os

K_FACTOR = 30

def expected_result(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def get_form(form_list):
    if len(form_list) == 0:
        return 7.0
    return sum(form_list) + (5 - len(form_list)) * 1.4

def train():
    print("Starting rigorous model training process...")
    matches_csv = 'matches_1930_2022.csv'
    
    if not os.path.exists(matches_csv):
        print(f"Error: {matches_csv} not found in the directory.")
        return

    print(f"Loading historical dataset from {matches_csv}...")
    df = pd.read_csv(matches_csv)
    
    # Filter missing scores and ensure Date is sorted
    df = df.dropna(subset=['home_score', 'away_score'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    elo_dict = {}
    form_dict = {}
    
    home_elo_list = []
    away_elo_list = []
    home_form_list = []
    away_form_list = []
    target_list = []
    
    print("Calculating running Elo ratings and Form...")
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        # Initialize if not exists
        if home not in elo_dict: elo_dict[home] = 1500
        if away not in elo_dict: elo_dict[away] = 1500
        if home not in form_dict: form_dict[home] = []
        if away not in form_dict: form_dict[away] = []
            
        # Capture pre-match states
        home_elo_list.append(elo_dict[home])
        away_elo_list.append(elo_dict[away])
        
        home_form_list.append(get_form(form_dict[home]))
        away_form_list.append(get_form(form_dict[away]))
        
        # Determine match outcome
        h_score = row['home_score']
        a_score = row['away_score']
        
        if h_score > a_score:
            target = 0
            h_points, a_points = 3, 0
            h_actual, a_actual = 1.0, 0.0
        elif h_score == a_score:
            target = 1
            h_points, a_points = 1, 1
            h_actual, a_actual = 0.5, 0.5
        else:
            target = 2
            h_points, a_points = 0, 3
            h_actual, a_actual = 0.0, 1.0
            
        target_list.append(target)
        
        # Update Elo
        e_home = expected_result(elo_dict[home], elo_dict[away])
        e_away = expected_result(elo_dict[away], elo_dict[home])
        
        elo_dict[home] = elo_dict[home] + K_FACTOR * (h_actual - e_home)
        elo_dict[away] = elo_dict[away] + K_FACTOR * (a_actual - e_away)
        
        # Update Form
        form_dict[home].append(h_points)
        form_dict[away].append(a_points)
        if len(form_dict[home]) > 5: form_dict[home].pop(0)
        if len(form_dict[away]) > 5: form_dict[away].pop(0)

    # Attach engineered features back to DataFrame
    df['home_elo'] = home_elo_list
    df['away_elo'] = away_elo_list
    df['home_form_last5'] = home_form_list
    df['away_form_last5'] = away_form_list
    df['is_home_advantage'] = (df['home_team'] == df['Host']).astype(int)
    df['target'] = target_list
    
    features = ['home_elo', 'away_elo', 'home_form_last5', 'away_form_last5', 'is_home_advantage']
    
    # 1. Time-Series Split
    print("Performing Time-Series split (Train: <= 2014, Test: >= 2018)...")
    train_df = df[df['Year'] <= 2014]
    test_df = df[df['Year'] >= 2018]
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Warning: Missing years in dataset. Falling back to simple 80/20 chronological split.")
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']

    # 2. Initialize HistGradientBoostingClassifier
    print("Initializing HistGradientBoostingClassifier...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        random_state=42
    )

    # 3. Train
    print("Training model...")
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    # 5. Export Model and State
    joblib.dump(model, 'model.pkl')
    print("Model successfully saved to model.pkl")
    
    team_stats = {'elo': elo_dict, 'form': form_dict}
    joblib.dump(team_stats, 'team_stats.pkl')
    print("Final team states successfully saved to team_stats.pkl")

if __name__ == '__main__':
    train()
