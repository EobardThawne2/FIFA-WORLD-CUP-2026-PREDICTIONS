from flask import Blueprint, render_template, jsonify
from predictor import get_match_probabilities, get_mock_schedule, generate_knockouts, generate_group_standings, get_win_chances, TEAM_GROUPS

# Create a Blueprint for our routes
TEAM_FLAGS = {
    "Argentina": "ar", "France": "fr", "Croatia": "hr", "Morocco": "ma",
    "United States": "us", "Mexico": "mx", "Canada": "ca", "Brazil": "br",
    "England": "gb-eng", "Spain": "es", "Germany": "de", "Portugal": "pt",
    "Netherlands": "nl", "Belgium": "be", "Uruguay": "uy", "Senegal": "sn",
    "Japan": "jp", "Korea Republic": "kr", "Australia": "au", "Switzerland": "ch",
    "Ecuador": "ec", "Tunisia": "tn", "Saudi Arabia": "sa", "IR Iran": "ir",
    "Czechia": "cz", "Czech Republic": "cz", "Türkiye": "tr", "Turkey": "tr",
    "Norway": "no", "Sweden": "se", "Scotland": "gb-sct", "Austria": "at",
    "Bosnia-Herzegovina": "ba", "Paraguay": "py", "Qatar": "qa", "Haiti": "ht",
    "Curaçao": "cw", "Côte d'Ivoire": "ci", "Egypt": "eg", "Cape Verde": "cv",
    "New Zealand": "nz", "Iraq": "iq", "Algeria": "dz", "Jordan": "jo",
    "Congo DR": "cd", "Ghana": "gh", "Panama": "pa", "Uzbekistan": "uz",
    "Colombia": "co"
}

def get_flag_html(team_name):
    code = TEAM_FLAGS.get(team_name)
    if code:
        return f'<img src="https://flagcdn.com/{code}.svg" class="country-flag" alt="">'
    return ""

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def fixtures():
    mock_schedule = get_mock_schedule()
    matches_with_probs = []
    
    for match in mock_schedule:
        probs = get_match_probabilities(match)
        match_data = match.copy()
        match_data["home"] = f"{get_flag_html(match['home'])} {match['home']}"
        match_data["away"] = f"{get_flag_html(match['away'])} {match['away']}"
        match_data["home_prob"] = probs[0]
        match_data["draw_prob"] = probs[1]
        match_data["away_prob"] = probs[2]
        matches_with_probs.append(match_data)
        
    # Get predicted winner
    groups = generate_group_standings()
    knockouts = generate_knockouts(groups)
    winner_name = knockouts["winner"]
    winner = f"{get_flag_html(winner_name)} {winner_name}"
        
    return render_template('fixtures.html', matches=matches_with_probs, winner=winner)

@main_bp.route('/chances')
def chances():
    raw_chances = get_win_chances()
    chances_data = []
    for rank, (team, pct) in enumerate(raw_chances, 1):
        chances_data.append({
            "rank": rank,
            "name": team,
            "flag_html": get_flag_html(team),
            "chance": round(pct, 2),
            "group": TEAM_GROUPS.get(team, "Unknown")
        })
    return render_template('chances.html', chances=chances_data)

@main_bp.route('/bracket')
def bracket():
    groups = generate_group_standings()
    knockouts = generate_knockouts(groups)
    
    for round_name, matches in knockouts.items():
        if round_name == "winner":
            knockouts[round_name] = f"{get_flag_html(matches)} {matches}"
            continue
        for m in matches:
            m['team1'] = f"{get_flag_html(m['team1'])} {m['team1']}"
            m['team2'] = f"{get_flag_html(m['team2'])} {m['team2']}"
            
    return render_template('bracket.html', knockouts=knockouts)

@main_bp.route('/api/predict-groups', methods=['POST'])
def predict_groups():
    groups = generate_group_standings()
    knockouts = generate_knockouts(groups)
    
    for g, teams in groups.items():
        for t in teams:
            t['team'] = f"{get_flag_html(t['team'])} {t['team']}"
            
    for round_name, matches in knockouts.items():
        if round_name == "winner":
            knockouts[round_name] = f"{get_flag_html(matches)} {matches}"
            continue
        for m in matches:
            m['team1'] = f"{get_flag_html(m['team1'])} {m['team1']}"
            m['team2'] = f"{get_flag_html(m['team2'])} {m['team2']}"

    return jsonify({
        "groups": groups,
        "knockouts": knockouts,
        "winner": knockouts["winner"]
    })
