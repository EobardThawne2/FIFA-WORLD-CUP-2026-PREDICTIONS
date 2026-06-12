import joblib
import random
import os
import pandas as pd

# Load model and state if they exist
MODEL_PATH = 'model.pkl'
STATS_PATH = 'team_stats.pkl'

model = None
team_stats = None

if os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH):
    model = joblib.load(MODEL_PATH)
    team_stats = joblib.load(STATS_PATH)
    print("Loaded model and team states successfully.")
else:
    print("model.pkl or team_stats.pkl not found. Falling back to dummy predictions.")

def get_form(form_list):
    if len(form_list) == 0:
        return 7.0
    return sum(form_list) + (5 - len(form_list)) * 1.4

TEAM_MAPPING = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey"
}

TEAM_GROUPS = {
    "Mexico": "A", "South Africa": "A", "Korea Republic": "A", "Czechia": "A",
    "Canada": "B", "Bosnia-Herzegovina": "B", "Qatar": "B", "Switzerland": "B",
    "Brazil": "C", "Morocco": "C", "Haiti": "C", "Scotland": "C",
    "United States": "D", "Paraguay": "D", "Australia": "D", "Türkiye": "D",
    "Curaçao": "E", "Ecuador": "E", "Germany": "E", "Côte d'Ivoire": "E",
    "Netherlands": "F", "Japan": "F", "Sweden": "F", "Tunisia": "F",
    "Belgium": "G", "Egypt": "G", "IR Iran": "G", "New Zealand": "G",
    "Spain": "H", "Cape Verde": "H", "Saudi Arabia": "H", "Uruguay": "H",
    "France": "I", "Senegal": "I", "Iraq": "I", "Norway": "I",
    "Argentina": "J", "Algeria": "J", "Austria": "J", "Jordan": "J",
    "Portugal": "K", "Congo DR": "K", "Uzbekistan": "K", "Colombia": "K",
    "England": "L", "Croatia": "L", "Ghana": "L", "Panama": "L"
}

# Precompute probabilities lookup table
PROB_LOOKUP = {}

def precompute_probabilities():
    global PROB_LOOKUP
    if not model or not team_stats:
        return
    schedule_file = 'schedule_2026.csv'
    if not os.path.exists(schedule_file):
        return
        
    try:
        df_sched = pd.read_csv(schedule_file)
        all_teams = list(set(df_sched['home_team'].dropna().unique()) | set(df_sched['away_team'].dropna().unique()))
        
        pairs = []
        for t1 in all_teams:
            for t2 in all_teams:
                if t1 != t2:
                    pairs.append((t1, t2))
                    
        home_list, away_list, h_elo_list, a_elo_list, h_form_list, a_form_list, is_home_adv_list = [], [], [], [], [], [], []
        hosts = ["United States", "Mexico", "Canada"]
        
        for home, away in pairs:
            home_std = TEAM_MAPPING.get(home, home)
            away_std = TEAM_MAPPING.get(away, away)
            is_home_adv = 1 if home_std in hosts else 0
            h_elo = team_stats['elo'].get(home_std, 1500)
            a_elo = team_stats['elo'].get(away_std, 1500)
            h_form = get_form(team_stats['form'].get(home_std, []))
            a_form = get_form(team_stats['form'].get(away_std, []))
            
            home_list.append(home)
            away_list.append(away)
            h_elo_list.append(h_elo)
            a_elo_list.append(a_elo)
            h_form_list.append(h_form)
            a_form_list.append(a_form)
            is_home_adv_list.append(is_home_adv)
            
        features_df = pd.DataFrame({
            'home_elo': h_elo_list,
            'away_elo': a_elo_list,
            'home_form_last5': h_form_list,
            'away_form_last5': a_form_list,
            'is_home_advantage': is_home_adv_list
        })
        
        probs = model.predict_proba(features_df)
        for i, (home, away) in enumerate(pairs):
            p = probs[i]
            home_win = round(p[0] * 100)
            draw = round(p[1] * 100)
            away_win = 100 - home_win - draw
            PROB_LOOKUP[(home, away)] = [home_win, draw, away_win]
            
        print(f"Precomputed {len(PROB_LOOKUP)} match probabilities successfully.")
    except Exception as e:
        print(f"Error precomputing probabilities: {e}")

# Run precomputation
precompute_probabilities()

def get_match_probabilities(match):
    """
    Returns match probabilities [Home_Win_%, Draw_%, Away_Win_%].
    """
    home = match['home']
    away = match['away']
    if (home, away) in PROB_LOOKUP:
        return PROB_LOOKUP[(home, away)]
        
    if model and team_stats:
        # Standardize names
        home = TEAM_MAPPING.get(home, home)
        away = TEAM_MAPPING.get(away, away)
        
        # Determine host advantage
        hosts = ["United States", "Mexico", "Canada"]
        is_home_adv = 1 if home in hosts else 0
        
        # Look up true Elo and Form (default to 1500 and average form 7 if new team)
        h_elo = team_stats['elo'].get(home, 1500)
        a_elo = team_stats['elo'].get(away, 1500)
        
        h_form = get_form(team_stats['form'].get(home, []))
        a_form = get_form(team_stats['form'].get(away, []))
        
        features_dict = {
            'home_elo': [h_elo], 
            'away_elo': [a_elo], 
            'home_form_last5': [h_form],
            'away_form_last5': [a_form],
            'is_home_advantage': [is_home_adv]
        }
        features_df = pd.DataFrame(features_dict)
        
        probs = model.predict_proba(features_df)[0]
        
        home_win = round(probs[0] * 100)
        draw = round(probs[1] * 100)
        away_win = 100 - home_win - draw
        
        return [home_win, draw, away_win]
    else:
        home_win = random.randint(10, 80)
        draw = random.randint(10, 100 - home_win)
        away_win = 100 - home_win - draw
        return [home_win, draw, away_win]

def parse_schedule():
    """Reads the real schedule CSV and returns a list of group stage matches."""
    schedule_file = 'schedule_2026.csv'
    if not os.path.exists(schedule_file):
        return []
        
    df = pd.read_csv(schedule_file)
    df = df[df['Round'] == 'Group stage']
    
    matches = []
    for index, row in df.iterrows():
        h = row['home_team']
        a = row['away_team']
        if pd.isna(h) or pd.isna(a): continue
        
        matches.append({
            "date": str(row['Date']),
            "group": TEAM_GROUPS.get(h, "Unknown"),
            "home": h,
            "away": a
        })
        
    return matches

def get_mock_schedule():
    return parse_schedule()

def generate_knockouts(groups):
    firsts = []
    seconds = []
    thirds = []
    
    # Extract teams by group letter order A-L
    for letter in 'ABCDEFGHIJKL':
        group_name = f"Group {letter}"
        teams = groups.get(group_name, [])
        if len(teams) >= 1: firsts.append(teams[0]["team"])
        if len(teams) >= 2: seconds.append(teams[1]["team"])
        if len(teams) >= 3: thirds.append(teams[2])
        
    thirds.sort(key=lambda x: (x["points"], x["gd"]), reverse=True)
    best_thirds = [t["team"] for t in thirds[:8]]
    
    round_32 = []
    for i in range(8):
        round_32.append({"team1": firsts[i], "team2": best_thirds[i]})
    for i in range(8, 12):
        round_32.append({"team1": firsts[i], "team2": seconds[i-8]})
    for i in range(4, 8):
        round_32.append({"team1": seconds[i], "team2": seconds[i+4]})
        
    def simulate_round(matches):
        next_round = []
        sim_matches = []
        for i in range(0, len(matches), 2):
            m1 = matches[i]
            p1 = get_match_probabilities({"home": m1["team1"], "away": m1["team2"]})
            w1 = m1["team1"] if p1[0] > p1[2] else m1["team2"]
            m1["advancing"] = 1 if w1 == m1["team1"] else 2
            sim_matches.append(m1)
            
            if i+1 < len(matches):
                m2 = matches[i+1]
                p2 = get_match_probabilities({"home": m2["team1"], "away": m2["team2"]})
                w2 = m2["team1"] if p2[0] > p2[2] else m2["team2"]
                m2["advancing"] = 1 if w2 == m2["team1"] else 2
                sim_matches.append(m2)
                next_round.append({"team1": w1, "team2": w2})
            else:
                next_round.append({"team1": w1, "team2": w1})
        return sim_matches, next_round

    r32, r16_input = simulate_round(round_32)
    r16, qf_input = simulate_round(r16_input)
    qf, sf_input = simulate_round(qf_input)
    sf, f_input = simulate_round(sf_input)
    
    final_match = f_input[0]
    pf = get_match_probabilities({"home": final_match["team1"], "away": final_match["team2"]})
    w_final = final_match["team1"] if pf[0] > pf[2] else final_match["team2"]
    final_match["advancing"] = 1 if w_final == final_match["team1"] else 2
    f_out = [final_match]
    
    return {
        "round_of_32": r32,
        "round_of_16": r16,
        "quarterfinals": qf,
        "semifinals": sf,
        "final": f_out,
        "winner": w_final
    }

def generate_group_standings():
    matches = parse_schedule()
    standings = {}
    
    for match in matches:
        g = match["group"]
        if g not in standings:
            standings[g] = {}
            
        home = match["home"]
        away = match["away"]
        
        if home not in standings[g]:
            standings[g][home] = {"points": 0, "gd": 0}
        if away not in standings[g]:
            standings[g][away] = {"points": 0, "gd": 0}
            
        probs = get_match_probabilities(match)
        max_prob = max(probs)
        
        if probs[0] == max_prob:
            standings[g][home]["points"] += 3
            standings[g][home]["gd"] += 1
            standings[g][away]["gd"] -= 1
        elif probs[2] == max_prob:
            standings[g][away]["points"] += 3
            standings[g][away]["gd"] += 1
            standings[g][home]["gd"] -= 1
        else:
            standings[g][home]["points"] += 1
            standings[g][away]["points"] += 1
            
    groups = {}
    for letter in 'ABCDEFGHIJKL':
        group_name = f"Group {letter}"
        if letter in standings:
            teams = standings[letter]
            team_list = []
            for team_name, stats in teams.items():
                team_list.append({
                    "team": team_name,
                    "points": stats["points"],
                    "gd": stats["gd"]
                })
            team_list.sort(key=lambda x: (x["points"], x["gd"]), reverse=True)
            groups[group_name] = team_list
        else:
            groups[group_name] = []
            
    return groups

WIN_CHANCES_CACHE = None

def get_win_chances(num_simulations=1000):
    global WIN_CHANCES_CACHE
    if WIN_CHANCES_CACHE is not None:
        return WIN_CHANCES_CACHE
        
    schedule_file = 'schedule_2026.csv'
    if not os.path.exists(schedule_file):
        return []
        
    # Build list of group stage matches as raw tuples for speed
    df_sched = pd.read_csv(schedule_file)
    group_stage_df = df_sched[df_sched['Round'] == 'Group stage']
    group_matches = []
    for _, row in group_stage_df.iterrows():
        h, a = row['home_team'], row['away_team']
        g = TEAM_GROUPS.get(h, "Unknown")
        group_matches.append((g, h, a))
        
    winners_count = {}
    for t in TEAM_GROUPS.keys():
        winners_count[t] = 0
        
    def sim_ko_round(matches):
        winners = []
        for home, away in matches:
            p = PROB_LOOKUP.get((home, away), [50, 0, 50])
            hw, aw = p[0], p[2]
            tot = hw + aw
            if tot == 0:
                tot = 100
                hw = 50
            r = random.randint(0, tot - 1)
            if r < hw:
                winners.append(home)
            else:
                winners.append(away)
        return winners
        
    for _ in range(num_simulations):
        # Reset standings
        standings = {g: {} for g in 'ABCDEFGHIJKL'}
        for t, g in TEAM_GROUPS.items():
            standings[g][t] = {"points": 0, "gd": 0}
            
        # Simulate group stage
        for g, home, away in group_matches:
            p = PROB_LOOKUP.get((home, away), [33, 34, 33])
            r = random.randint(0, 99)
            if r < p[0]:
                standings[g][home]["points"] += 3
                standings[g][home]["gd"] += 1
                standings[g][away]["gd"] -= 1
            elif r < p[0] + p[1]:
                standings[g][home]["points"] += 1
                standings[g][away]["points"] += 1
            else:
                standings[g][away]["points"] += 3
                standings[g][away]["gd"] += 1
                standings[g][home]["gd"] -= 1
                
        # Group standings sorting
        firsts = []
        seconds = []
        thirds = []
        for g in 'ABCDEFGHIJKL':
            teams = list(standings[g].items())
            teams.sort(key=lambda x: (x[1]["points"], x[1]["gd"]), reverse=True)
            firsts.append(teams[0][0])
            seconds.append(teams[1][0])
            thirds.append((teams[2][0], teams[2][1]["points"], teams[2][1]["gd"]))
            
        thirds.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_thirds = [t[0] for t in thirds[:8]]
        
        # Round of 32
        r32_matches = []
        for i in range(8):
            r32_matches.append((firsts[i], best_thirds[i]))
        for i in range(8, 12):
            r32_matches.append((firsts[i], seconds[i-8]))
        for i in range(4, 8):
            r32_matches.append((seconds[i], seconds[i+4]))
            
        # R32 -> R16
        r16_teams = sim_ko_round(r32_matches)
        # R16 -> QF
        qf_matches = [(r16_teams[i], r16_teams[i+1]) for i in range(0, 16, 2)]
        qf_teams = sim_ko_round(qf_matches)
        # QF -> SF
        sf_matches = [(qf_teams[i], qf_teams[i+1]) for i in range(0, 8, 2)]
        sf_teams = sim_ko_round(sf_matches)
        # SF -> F
        f_matches = [(sf_teams[0], sf_teams[1]), (sf_teams[2], sf_teams[3])]
        f_teams = sim_ko_round(f_matches)
        # Final
        winner = sim_ko_round([(f_teams[0], f_teams[1])])[0]
        
        winners_count[winner] = winners_count.get(winner, 0) + 1
        
    sorted_winners = sorted(
        [(team, count / num_simulations * 100) for team, count in winners_count.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    WIN_CHANCES_CACHE = sorted_winners
    return WIN_CHANCES_CACHE

