import zipfile
import os
import shutil
import json
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

this_file_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

json_dir = this_file_dir + "data/raw/cricksheet/json/"
counter = 0
counter_lock = Lock()

#################################
# FILE FIND: data_download.py
#################################
def execute_scraper():
    json_url = "https://cricsheet.org/downloads/all_json.zip"
    people_csv_url = "https://cricsheet.org/register/people.csv"

    this_file_dir = os.path.dirname(os.path.abspath(__file__))  # Ensure this is defined
    target_json_dir = os.path.join(this_file_dir, 'data/raw/cricksheet/json/')
    target_people_csv_path = os.path.join(this_file_dir, "data/raw/cricksheet/people.csv")

    json_zip_file = "all_json.zip"

    if not os.path.exists(target_json_dir):
        os.makedirs(target_json_dir)

    # Download CSV file
    response = requests.get(people_csv_url)
    with open(target_people_csv_path, 'w') as file:
        file.write(response.text)

    # Download JSON ZIP file
    response = requests.get(json_url, stream=True)
    with open(json_zip_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    # Unzip JSON files
    with zipfile.ZipFile(json_zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_json_dir)

    # Remove ZIP file
    os.remove(json_zip_file)

#################################
# FILE FIND: json_generator.py
#################################
dir_with_cricsheets_json = json_dir
stored_dir = dir_with_cricsheets_json

output_csv_json_generator = this_file_dir + "data/interim/total_data.csv"

match_types_unique = set()


match_attributes = [
    "match_id",
    "gender",
    "balls_per_over",
    "date",
    "series_name",
    "match_type"
]

batsman_attributes = [
    "player_id",
    "runs_scored",
    "player_out", 
    "balls_faced",
    "fours_scored",
    "sixes_scored",
    "catches_taken",
    "run_out_direct",
    "run_out_throw",
    "stumpings_done",
    "out_kind",
    "dot_balls_as_batsman",
    "order_seen",
    "balls_bowled", 
    "runs_conceded",
    "wickets_taken",
    "bowled_done",
    "lbw_done",
    "maidens",
    "dot_balls_as_bowler",
    "player_team",
    "opposition_team"
]

bowler_attributes = [
    "catches_taken",
    "run_out_direct",
    "run_out_throw",
    "balls_bowled", 
    "runs_conceded",
    "wickets_taken",
    "catches_taken",
    "bowled_done",
    "lbw_done",
    "maidens",
    "dot_balls_as_bowler"
]

def import_data(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
            
    
    info_data = json_data.get("info", {})
    info_data["match_id"] = file_path.split("/")[-1].split(".")[0]
    innings_data = json_data.get("innings", [])
    
    
    return info_data, innings_data

def fix_dates(dates):
    ret = ""
    if len(dates) > 1:
        ret = dates[0] + " - " + dates[1]
    else:
        ret = dates[0]
    return ret

def parse_info_data(info_data):
    dates = info_data["dates"]
    dates = fix_dates(dates)
    info_data["date"] = dates
    try:
        info_data["series_name"] = info_data["event"]["name"].replace(",", " ")
    except:
        info_data["series_name"] = "None"
    ret = {}

    match_type = info_data["match_type"]

    if match_type not in match_types_unique:
        match_types_unique.add(match_type)

    for attribute in match_attributes:
        ret[attribute] = info_data.get(attribute, None)
    return ret

def get_players_data_dict(players_in_match, player_ids, info_dict):
    total_data = {}
    
    for player in players_in_match:
        total_data[player] = {}
        player_id = player_ids[player]
        for attribute in batsman_attributes:
            total_data[player][attribute] = 0
        for attribute in bowler_attributes:
            total_data[player][attribute] = 0
        total_data[player]["player_id"] = player_id
    
    team_players_mapping = info_dict["players"]
    team_1 = list(team_players_mapping.keys())[0]
    team_2 = list(team_players_mapping.keys())[1]

    for player in team_players_mapping[team_1]:
        total_data[player]["player_team"] = team_1
        total_data[player]["opposition_team"] = team_2
    for player in team_players_mapping[team_2]:
        total_data[player]["player_team"] = team_2
        total_data[player]["opposition_team"] = team_1

    return total_data

def get_overs(session):
    try:
        overs = session["overs"]
    except:
        overs = []
    return overs

def is_wicket(ball):
    try:
        wicket = ball["wickets"]
        return True
    except:
        return False

def get_wicket_data(wicket):
    ret = {}
    wicket = wicket[0]
    ret["player_out"] = wicket["player_out"]
    ret["kind"] = wicket["kind"]
    try:
        ret["fielders"] = wicket["fielders"]
    except:
        ret["fielders"] = None
    return ret

def split_data(total_data):
    batsmen = []
    bowlers = []

    for player in total_data:
        batsmen.append(player)
        bowlers.append(player)

    batsmen_data = {}
    bowlers_data = {}

    for player in batsmen:
        batsmen_data[player] = {}
        for attribute in batsman_attributes:
            batsmen_data[player][attribute] = total_data[player][attribute]
    
    for player in bowlers:
        bowlers_data[player] = {}
        for attribute in bowler_attributes:
            bowlers_data[player][attribute] = total_data[player][attribute]
    return batsmen_data, bowlers_data


def export_to_csv(batsmen, bowlers, match_attributes_parsed, batsman_order):
    if not os.path.exists(output_csv_json_generator):
        with open(output_csv_json_generator, 'w') as f:
            for i in match_attributes:
                f.write(i + ',')
            f.write("name,")
            for i in range(len(batsman_attributes)-1):
                f.write(batsman_attributes[i] + ',')
            f.write(batsman_attributes[-1] + '\n')

    for batsman in batsmen:
        try:
            batsmen[batsman]["order_seen"] = batsman_order.index(batsman) + 1
        except:
            pass
        
    with open(output_csv_json_generator, 'a') as f:
        for player in bowlers:
            for i in match_attributes_parsed:
                f.write(str(match_attributes_parsed[i]) + ',')
            f.write(player + ',')
            for i in range(len(batsman_attributes[:-1])):
                try:
                    to_write = batsmen[player][batsman_attributes[i]]
                except:
                    to_write = 0
                f.write(str(to_write) + ',')
            f.write(batsmen[player][batsman_attributes[-1]] + '\n')


def parse_innings_data(innings_data, players_in_match, match_attributes_parsed, player_ids, match_info):
    for session in innings_data:
        total_data = get_players_data_dict(players_in_match, player_ids, match_info)
        batsman_order = []
        num_seen = 1

        overs = get_overs(session)
        for over in overs:
            over_ball_list = over["deliveries"]
            runs_in_over = 0
            for ball in over_ball_list:
                batsman = ball["batter"]
                bowler = ball["bowler"]

                if batsman not in batsman_order:
                    total_data[batsman]["order_seen"] = num_seen
                    num_seen += 1
                    batsman_order.append(batsman)

                runs_scored = ball["runs"]["batter"]
                extras = ball["runs"]["extras"]
                runs = runs_scored + extras
                runs_in_over += runs

                if is_wicket(ball):
                    wicket_data = get_wicket_data(ball["wickets"])

                    kind = wicket_data["kind"]

                    total_data[bowler]["wickets_taken"] += 1

                    if kind == "caught":
                        for fielder in wicket_data["fielders"]:
                            try:
                                fielder = fielder["name"]
                            except:
                                fielder = None
                            try:
                                total_data[fielder]["catches_taken"] += 1
                            except:
                                pass
                    
                    if kind == "run out":
                        total_data[bowler]["wickets_taken"] -= 1
                        fielders = wicket_data["fielders"]
                        fielders = fielders if fielders != None else []

                        if fielders == []:
                            pass
                        else:
                            if len(fielders) >= 2:
                                for fielder in fielders:
                                    try:
                                        total_data[fielder["name"]]["run_out_throw"] += 1
                                    except:
                                        pass
                            else:
                                try:
                                    total_data[fielders[0]["name"]]["run_out_direct"] += 1
                                except:
                                    pass

                    if kind == "stumped":
                        total_data[bowler]["wickets_taken"] -= 1
                        fielders = wicket_data["fielders"]
                        try:
                            total_data[fielders[0]["name"]]["stumpings_done"] += 1
                        except:
                            pass

                    if kind == "bowled":
                        total_data[bowler]["bowled_done"] += 1
                    if kind == "lbw":
                        total_data[bowler]["lbw_done"] += 1
                    total_data[batsman]["out_kind"] = kind
                    total_data[batsman]["player_out"] = 1
                
                    total_data[batsman]["player_out"] = 1
                if not runs_scored:
                    total_data[bowler]["dot_balls_as_bowler"] += 1
                    total_data[batsman]["dot_balls_as_batsman"] += 1

                total_data[bowler]["runs_conceded"] += runs
                total_data[bowler]["balls_bowled"] += 1
                total_data[batsman]["runs_scored"] += runs_scored
                total_data[batsman]["balls_faced"] += 1
                if runs_scored == 4:
                    total_data[batsman]["fours_scored"] += 1
                if runs_scored == 6:
                    total_data[batsman]["sixes_scored"] += 1
            if runs_in_over == 0:
                total_data[bowler]["maidens"] += 1
        batsmen, bowlers = split_data(total_data)
        export_to_csv(batsmen, bowlers, match_attributes_parsed, batsman_order)
    return batsman_attributes, bowler_attributes

def get_players(info_data):
    players = []
    for i in list(info_data.get("players", {}).values()):
        for player in i:
            players.append(player)
    return players

def generate(file_path):
    info_data, innings_data = import_data(file_path)

    player_ids = info_data["registry"]["people"]

    team_split = info_data["players"]
    players_in_match = get_players(info_data)

    match_attributes_parsed = parse_info_data(info_data)
    parse_innings_data(innings_data, players_in_match, match_attributes_parsed, player_ids, info_data)


def json_generator():
    ignore_files = [".", "..", ".DS_Store", "README.txt"]

    total_files = len(os.listdir(stored_dir))
    files_done = 0
    for file in os.listdir(stored_dir):

        files_done += 1
        print(f"Processing file {files_done}/{total_files}")
        print("File: ", file)
        if file not in ignore_files:
            generate(stored_dir + file)
    return 1

#################################
# FILE FIND: mw_overall.py
#################################
output_csv_overall_path = this_file_dir + "data/interim/mw_overall.csv"

attributes_overall = [
    'match_id',
    'innings',
    'batting_team',
    'runs_off_bat',
    'extras',
    'wides',
    'noballs',
    'byes',
    'legbyes',
    'penalty',
    'player_dismissed',
    'bowled',
    'caught',
    'caught and bowled',
    'lbw',
    'run out',
    'balls_per_over',
    'city',
    'dates',
    'event_name',
    'match_number',
    'gender',
    'match_type',
    'match_type_number',
    'match_referees',
    'tv_umpires',
    'umpires',
    'result',
    'player_of_match',
    'season', 'team_type','teams',
    'toss_decision','toss_winner', 'venue','winner',
    'players', 'stumped',
    'hit wicket', 'retired hurt', 'retired not out',
    'obstructing the field','retired out',
    'handled the ball', 'hit the ball twice'
]

def import_data_overall(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
            
    
    info_data = json_data.get("info", {})
    info_data["match_id"] = file_path.split("/")[-1].split(".")[0]
    innings_data = json_data.get("innings", [])
    
    return info_data, innings_data

def get_wicket_type(wicket):
    wicket = wicket[0]
    return wicket["kind"]

def export_dict(dict_overall):
    if not os.path.exists(output_csv_overall_path):
        with open(output_csv_overall_path, 'w') as f:
            print("File created")
            for i in attributes_overall[:-1]:
                f.write(i + ',')
            f.write(attributes_overall[-1] + '\n')

    with open(output_csv_overall_path, 'a') as f:
        for i in range(len(attributes_overall)-1):
            f.write(str(dict_overall[attributes_overall[i]]) + ',')
        f.write(str(dict_overall[attributes_overall[-1]]) + '\n')

def parse_innings_data_overall(innings_data, match_info):
    for i, session in enumerate(innings_data):
        innings_number = i + 1

        export = dict()
        export["match_id"] = match_info["match_id"]
        export["innings"] = innings_number
        export["batting_team"] = session["team"]
        export["runs_off_bat"] = 0
        export["extras"] = 0
        export["wides"] = 0
        export["noballs"] = 0
        export["byes"] = 0
        export["legbyes"] = 0
        export["penalty"] = 0
        export["player_dismissed"] = 0
        export["bowled"] = 0
        export["caught"] = 0
        export["caught and bowled"] = 0
        export["lbw"] = 0
        export["run out"] = 0
        export["balls_per_over"] = match_info['balls_per_over']
        try:    
            export["city"] = match_info["city"]
        except:
            export["city"] = ""
        export["dates"] = '"' + str(", ".join(match_info["dates"])) + '"'
        try:
            export['event_name'] = str(match_info['event']['name']).replace(',', ' ')
        except:
            export['event_name'] = ""
        try:
            export['match_number'] = match_info['event']['match_number']
        except:
            export['match_number'] = ""
        export["gender"] = match_info["gender"]
        export["match_type"] = match_info["match_type"]
        try:
            export["match_type_number"] = match_info["match_type_number"]
        except:
            export["match_type_number"] = ""
        try:
            export["match_referees"] = '"' + str(", ".join(match_info["officials"]["match_referees"])) + '"'
        except:
            export["match_referees"] = ""
        try:
            export["tv_umpires"] = '"' + str(", ".join(match_info["officials"]["tv_umpires"])) + '"'
        except:
            export["tv_umpires"] = ""
        try:
            export["umpires"] = '"' + str(", ".join(match_info["officials"]["umpires"])) + '"'
        except:
            export["umpires"] = ""
        try:
            export["result"] = match_info["outcome"]["result"]
        except:
            export["result"] = ""
        try:
            export["player_of_match"] = str(" ".join(match_info["player_of_match"]))
        except:
            export["player_of_match"] = ""

        export["season"] = match_info["season"]
        export["team_type"] = match_info["team_type"]
        export["teams"] = '"' + ", ".join(match_info["teams"]) + '"'
        export["toss_decision"] = match_info["toss"]["decision"]
        export["toss_winner"] = match_info["toss"]["winner"]
        export["venue"] = str(match_info["venue"]).replace(',', ' ')
        if export["result"] == "":
            export["winner"] = match_info['outcome']['winner']
        else:
            export["winner"] = export["result"]
        players = list(match_info['players'].values())
        players_export = players[0] + players[1]
        export["players"] = '"' + ", ".join(players_export) + '"'
        export["stumped"] = 0
        export["hit wicket"] = 0
        export["retired hurt"] = 0
        export["retired not out"] = 0
        export["obstructing the field"] = 0
        export["retired out"] = 0
        export["handled the ball"] = 0
        export["hit the ball twice"] = 0


        overs = get_overs(session)
        for over in overs:
            over_ball_list = over["deliveries"]
            for ball in over_ball_list:
                runs_scored = ball["runs"]["batter"]
                runs_extras = ball["runs"]["extras"]

                if "extras" in ball:
                    extras = ball["extras"]
                    for extra in extras:
                        if extra == "wides":
                            export["wides"] += ball["extras"]["wides"]
                        elif extra == "noballs":
                            export["noballs"] += 1
                        elif extra == "byes":
                            export["byes"] += ball["extras"]["byes"]
                        elif extra == "legbyes":
                            export["legbyes"] += ball["extras"]["legbyes"]
                        elif extra == "penalty":
                            export["penalty"] += ball["extras"]["penalty"]
                export["runs_off_bat"] += runs_scored
                export["extras"] += runs_extras

                if is_wicket(ball):
                    export["player_dismissed"] += 1
                    wicket = ball["wickets"]
                    wicket_type = get_wicket_type(wicket)
                    if wicket_type == "bowled":
                        export["bowled"] += 1
                    elif wicket_type == "caught":
                        export["caught"] += 1
                    elif wicket_type == "caught and bowled":
                        export["caught and bowled"] += 1
                    elif wicket_type == "lbw":
                        export["lbw"] += 1
                    elif wicket_type == "run out":
                        export["run out"] += 1
                    elif wicket_type == "stumped":
                        export["stumped"] += 1
                    elif wicket_type == "hit wicket":
                        export["hit wicket"] += 1
                    elif wicket_type == "retired hurt":
                        export["retired hurt"] += 1
                    elif wicket_type == "retired not out":
                        export["retired not out"] += 1
                    elif wicket_type == "obstructing the field":
                        export["obstructing the field"] += 1
                    elif wicket_type == "retired out":
                        export["retired out"] += 1
                    elif wicket_type == "handled the ball":
                        export["handled the ball"] += 1
                    elif wicket_type == "hit the ball twice":
                        export["hit the ball twice"] += 1
        export_dict(export)


def get_players_overall(info_data):
    players = []
    for i in list(info_data.get("players", {}).values()):
        for player in i:
            players.append(player)
    return players

def generate_overall(file_path):
    info_data, innings_data = import_data_overall(file_path)

    parse_innings_data_overall(innings_data, info_data)


def mw_overall_generator():
    ignore_files = [".", "..", ".DS_Store", "README.txt"]

    total_files = len(os.listdir(stored_dir))
    files_done = 0
    for file in os.listdir(stored_dir):
        files_done += 1
        print(f"Processing file {files_done}/{total_files}")
        print("File: ", file)
        if file not in ignore_files:
            generate_overall(stored_dir + file)
    return 1


#################################
# FILE FIND: adding_names.py
#################################
def get_player_details(cricinfo_id, total_players):
    global counter
    url = f"https://www.espncricinfo.com/cricketers/player-{cricinfo_id}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        full_name = batting_style = bowling_style = playing_role = None
        teams = []
        full_name_section = soup.find('div', class_="ds-col-span-2 lg:ds-col-span-1")
        if full_name_section:
            name_label = full_name_section.find('p', class_="ds-text-tight-m ds-font-regular ds-uppercase ds-text-typo-mid3")
            if name_label and name_label.text == "Full Name":
                full_name = full_name_section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()

        info_sections = soup.find_all('div')
        for section in info_sections:
            label = section.find('p', class_="ds-text-tight-m ds-font-regular ds-uppercase ds-text-typo-mid3")
            if label:
                if label.text == "Batting Style":
                    batting_style = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
                elif label.text == "Bowling Style":
                    bowling_style = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
                elif label.text == "Playing Role":
                    playing_role = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()

        teams_section = soup.find('div', class_="ds-grid lg:ds-grid-cols-3 ds-grid-cols-2 ds-gap-y-4")
        if teams_section:
            team_links = teams_section.find_all('a', class_="ds-flex ds-items-center ds-space-x-4")
            for team_link in team_links:
                title = team_link.get('title', '')
                team_name = title.split("'s ", 1)[1].strip()
                if team_name.endswith(" team profile"):
                    team_name = team_name[:-13]
                if team_name:
                    teams.append(team_name)
        with counter_lock:
            counter += 1
            print(f"Progress: {counter}/{total_players} players processed.")

        return cricinfo_id, full_name, batting_style, bowling_style, playing_role, teams

    else:
        with counter_lock:
            counter += 1
            print(f"Progress: {counter}/{total_players} players processed (Failed).")

        return cricinfo_id, None, None, None, None, []


def run_scraper_parallel(data, max_workers):
    total_players = len(data)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(get_player_details, row['key_cricinfo'], total_players): row['key_cricinfo']
            for _, row in data.iterrows()
        }

        for future in as_completed(future_to_id):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error occurred: {e}")

    return results

def aggregate():
    total_data = pd.read_csv(this_file_dir + "data/interim/total_data.csv")
    total_data["match_id"] = total_data["match_id"].astype(str)
    total_data["player_id"] = total_data["player_id"].astype(str)


    agg_dict = {
        'gender': 'first',
        'balls_per_over': 'first',
        'start_date': 'first',
        'series_name': 'first',
        'match_type': 'first',
        'name': 'first',
        'runs_scored': 'sum',
        'player_out': 'sum',
        'balls_faced': 'sum',
        'fours_scored': 'sum',
        'sixes_scored': 'sum',
        'catches_taken': 'sum',
        'run_out_direct': 'sum',
        'run_out_throw': 'sum',
        'stumpings_done': 'sum',
        'out_kind': 'first',
        'dot_balls_as_batsman': 'sum',
        'order_seen': 'first',
        'balls_bowled': 'sum',
        'runs_conceded': 'sum',
        'wickets_taken': 'sum',
        'bowled_done': 'sum',
        'lbw_done': 'sum',
        'maidens': 'sum',
        'dot_balls_as_bowler': 'sum',
        'player_team': 'first',
        'opposition_team': 'first'
    }

    result = total_data.groupby(['player_id', 'match_id']).agg(agg_dict).reset_index()
    result.to_csv(this_file_dir + 'data/interim/mw_pw.csv')



def rename_date():
    total_data_path = this_file_dir + "data/interim/total_data.csv"
    df = pd.read_csv(total_data_path, index_col= False)

    df['date'] = df['date'].str.split(" - ").str[0]
    df.rename(columns={'date': 'start_date'}, inplace=True)
    os.remove(total_data_path)
    df.to_csv(total_data_path)




def get_player_details(cricinfo_id, total_players):
    global counter
    cricinfo_id = int(cricinfo_id)
    print(f"Processing player with cricinfo_id: {cricinfo_id}")
    url = f"https://www.espncricinfo.com/cricketers/player-{cricinfo_id}"
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    }
    response = requests.get(url, headers=headers)
    # print(response.status_code)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        full_name = batting_style = bowling_style = playing_role = None
        teams = []
        
        full_name_section = soup.find('div', class_="ds-col-span-2 lg:ds-col-span-1")
        if full_name_section:
            name_label = full_name_section.find('p', class_="ds-text-tight-m ds-font-regular ds-uppercase ds-text-typo-mid3")
            if name_label and name_label.text == "Full Name":
                full_name = full_name_section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
        
        info_sections = soup.find_all('div')
        for section in info_sections:
            label = section.find('p', class_="ds-text-tight-m ds-font-regular ds-uppercase ds-text-typo-mid3")
            if label:
                if label.text == "Batting Style":
                    batting_style = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
                elif label.text == "Bowling Style":
                    bowling_style = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
                elif label.text == "Playing Role":
                    playing_role = section.find('span', class_="ds-text-title-s ds-font-bold ds-text-typo").text.strip()
        
        teams_section = soup.find('div', class_="ds-grid lg:ds-grid-cols-3 ds-grid-cols-2 ds-gap-y-4")
        if teams_section:
            team_links = teams_section.find_all('a', class_="ds-flex ds-items-center ds-space-x-4")
            for team_link in team_links:
                title = team_link.get('title', '')
                team_name = title.split("'s ", 1)[1].strip()
                if team_name.endswith(" team profile"):
                    team_name = team_name[:-13]
                if team_name:
                    teams.append(team_name)

        with counter_lock:
            counter += 1
            print(f"Progress: {counter}/{total_players} players processed.")  
        # print(full_name, batting_style, bowling_style, playing_role, teams)         
        return cricinfo_id, full_name, batting_style, bowling_style, playing_role, teams
    else:
        return cricinfo_id, None, None, None, None, []

def run_scraper_parallel(data, total_players, max_workers):
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(get_player_details, row['key_cricinfo'], total_players): row['key_cricinfo']
            for _, row in data.iterrows()
        }
        
        for future in as_completed(future_to_id):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error occurred: {e}")
    
    return results



def adding_names():
    global counter

    data = pd.read_csv(this_file_dir + 'data/raw/cricksheet/people.csv')
    total_players = len(data)
    counter = 0 
    # print(data)
    scraped_data = run_scraper_parallel(data, total_players, max_workers=50)

    scraped_df = pd.DataFrame(scraped_data, columns=['key_cricinfo', 'full_name', 'batting_style', 'bowling_style', 'playing_role', 'teams'])

    data = data.merge(scraped_df, on='key_cricinfo', how='left')

    data = data.rename(columns={"identifier": "player_id"})

    input_data = pd.read_csv(this_file_dir + 'data/interim/mw_pw.csv')
    final_data = input_data.merge(data, on='player_id', how='left')

    final_data.to_csv(this_file_dir + 'data/interim/mw_pw_profiles.csv', index=False)

    print("Player data updated successfully with parallel scraping.")
    return final_data

#################################
# FILE FIND: style_features.py
#################################
spin = [
    'Legbreak Googly',
    'Right arm Offbreak, Legbreak Googly',
    'Slow Left arm Orthodox',
    'Right arm Offbreak, Slow Left arm Orthodox',
    'Slow Left arm Orthodox, Slow Left arm Orthodox',
    'Slow Left arm Orthodox, Left arm Wrist spin',
    'Right arm Offbreak',
    'Right arm Offbreak, Legbreak',
    'Right arm Offbreak, Slow Left arm Orthodox',
    'Right arm Offbreak, Legbreak Googly',

    ]

right_fast = [
    'Right arm Fast',
    'Right arm Fast medium',
    'Right arm Fast medium, Right arm Medium'
]

left_fast = [
    'Left arm Fast',
    'Left arm Fast medium'
]

attributes = [
    'balls_against_spin',
    'balls_against_right_fast',
    'balls_against_left_fast',

    'runs_against_spin',
    'runs_against_right_fast',
    'runs_against_left_fast',

    'outs_against_spin',
    'outs_against_right_fast',
    'outs_against_left_fast',

    'balls_against_left',
    'balls_against_right',

    'runs_conceeded_against_left',
    'runs_conceeded_against_right',

    'wickets_against_left',
    'wickets_against_right'
]

style_features_output = this_file_dir + "data/interim/style_based_features.csv"
json_dir = this_file_dir + 'data/raw/cricksheet/json/'
mw_pw_profiles_path = this_file_dir + 'data/interim/mw_pw_profiles.csv'



def import_data_style(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
            
    
    info_data = json_data.get("info", {})
    info_data["match_id"] = file_path.split("/")[-1].split(".")[0]
    innings_data = json_data.get("innings", [])
    
    return info_data, innings_data


def get_players_style(info_data):
    players = []
    for i in list(info_data.get("players", {}).values()):
        for player in i:
            players.append(player)
    return players

def prep_dicts(name_id):
    ret = {}

    for name in name_id:
        ret[name] = {}
        for attribute in attributes:
            ret[name][attribute] = 0
    return ret

def get_overs(session):
    try:
        overs = session["overs"]
    except:
        overs = []
    return overs

def is_wicket(ball):
    try:
        wicket = ball["wickets"]
        return True
    except:
        return False

def export_dict_to_csv(player_data, file_path):
    match_id = file_path.split("/")[-1].split(".")[0]
    if not os.path.exists(style_features_output):
        with open(style_features_output, 'w') as f:
            f.write(f"name,match_id,")
            for i in range(len(attributes)-1):
                f.write(attributes[i] + ',')
            f.write(attributes[-1] + '\n')

    with open(style_features_output, 'a') as f:
        for player in player_data:
            f.write(f"{player},{match_id},")
            for i in range(len(attributes)-1):
                f.write(str(player_data[player][attributes[i]]) + ',')
            f.write(str(player_data[player][attributes[-1]]) + '\n')



def parse_innings_data_style(innings_data, name_id, file_path):
    for session in innings_data:
        player_data = prep_dicts(name_id)

        overs = get_overs(session)
        for over in overs:
            over_ball_list = over["deliveries"]
            for ball in over_ball_list:
                batsman = ball["batter"]
                bowler = ball["bowler"]

                bowler_id = name_id[bowler]
                batsman_id = name_id[batsman]

                batsman_style = id_batting_style[batsman_id][0]
                bowler_style = id_bowling_style[bowler_id][0]

                runs_scored = ball["runs"]["batter"]
                extras = ball["runs"]["extras"]
                runs = runs_scored + extras

                if bowler_style in spin:
                    if batsman_style == "Right hand Bat":
                        player_data[batsman]["balls_against_spin"] += 1
                        player_data[batsman]["runs_against_spin"] += runs

                        player_data[bowler]["balls_against_right"] += 1
                        player_data[bowler]["runs_conceeded_against_right"] += runs

                        if is_wicket(ball):
                            player_data[batsman]["outs_against_spin"] += 1
                            player_data[bowler]["wickets_against_right"] += 1
                    
                    elif batsman_style == "Left hand Bat":
                        player_data[batsman]["balls_against_spin"] += 1
                        player_data[batsman]["runs_against_spin"] += runs

                        player_data[bowler]["balls_against_left"] += 1
                        player_data[bowler]["runs_conceeded_against_left"] += runs

                        if is_wicket(ball):
                            player_data[batsman]["outs_against_spin"] += 1
                            player_data[bowler]["wickets_against_left"] += 1
                
                elif bowler_style in left_fast:
                    if batsman_style == "Right hand Bat":
                        player_data[batsman]["balls_against_left_fast"] += 1
                        player_data[batsman]["runs_against_left_fast"] += runs

                        player_data[bowler]["balls_against_left"] += 1
                        player_data[bowler]["runs_conceeded_against_left"] += runs

                        if is_wicket(ball):
                            player_data[batsman]["outs_against_left_fast"] += 1
                            player_data[bowler]["wickets_against_left"] += 1
                    
                    elif batsman_style == "Left hand Bat":
                        player_data[batsman]["balls_against_left_fast"] += 1
                        player_data[batsman]["runs_against_left_fast"] += runs

                        player_data[bowler]["balls_against_right"] += 1
                        player_data[bowler]["runs_conceeded_against_right"] += runs

                        if is_wicket(ball):
                            player_data[batsman]["outs_against_left_fast"] += 1
                            player_data[bowler]["wickets_against_right"] += 1
                
                elif bowler_style in right_fast:
                    if batsman_style == "Right hand Bat":
                        player_data[batsman]["balls_against_right_fast"] += 1
                        player_data[batsman]["runs_against_right_fast"] += runs

                        player_data[bowler]["balls_against_right"] += 1
                        player_data[bowler]["runs_conceeded_against_right"] += runs

                        if is_wicket(ball):
                            player_data[batsman]["outs_against_right_fast"] += 1
                            player_data[bowler]["wickets_against_right"] += 1

                    elif batsman_style == "Left hand Bat":
                        player_data[batsman]["balls_against_right_fast"] += 1
                        player_data[batsman]["runs_against_right_fast"] += runs

                        player_data[bowler]["balls_against_left"] += 1
                        player_data[bowler]["runs_conceeded_against_left"] += runs

                        if is_wicket(ball):
                            player_data[batsman]["outs_against_right_fast"] += 1
                            player_data[bowler]["wickets_against_left"] += 1
        export_dict_to_csv(player_data, file_path)

def generate_style(file_path):
    info_data, innings_data = import_data_style(file_path)

    name_id = info_data['registry']['people']

    players_in_match = get_players_style(info_data)

    for player in players_in_match:
        if player not in name_id:
            name_id[player] = None

    parse_innings_data_style(innings_data, name_id, file_path)

def style_based_features():
    ignore_files = ['.', '..', '.DS_Store', 'README.txt']
    total_files = 0
    files_processed = 0

    global mw_pw_profiles
    global id_batting_style
    global id_bowling_style
    global id_playing_role

    mw_pw_profiles = pd.read_csv(mw_pw_profiles_path, index_col = False)

    parent_df = mw_pw_profiles[['player_id', 'batting_style', 'bowling_style', 'playing_role']]
    id_bowling_style = parent_df[['player_id', 'bowling_style']]
    id_batting_style = parent_df[['player_id', 'batting_style']]
    id_playing_role = parent_df[['player_id', 'playing_role']]

    id_bowling_style.drop_duplicates(subset=['player_id'], keep='first', inplace=True)
    id_batting_style.drop_duplicates(subset=['player_id'], keep='first', inplace=True)
    id_playing_role.drop_duplicates(subset=['player_id'], keep='first', inplace=True)

    id_bowling_style = id_bowling_style.set_index('player_id').T.to_dict('list')
    id_batting_style = id_batting_style.set_index('player_id').T.to_dict('list')
    id_playing_role = id_playing_role.set_index('player_id').T.to_dict('list')

    for file in os.listdir(json_dir):
        files_processed += 1
        print(f"Processing file {files_processed}/{total_files}")
        print(f"File: {file}")
        if file not in ignore_files:
            generate_style(json_dir + file)

    df = pd.read_csv(style_features_output, index_col = False)
    df = df.groupby(['match_id', 'name']).sum()
    os.remove(style_features_output)
    df.to_csv(style_features_output)
    return 1


def integrate_sample_data():
    # if files exist in this_file_dir + '../out_of_sample_data/'
    # then move them to this_file_dir + 'data/raw/cricksheet/json/'

    out_of_sample_dir = this_file_dir + '../out_of_sample_data/'
    cricksheet_dir = this_file_dir + 'data/raw/cricksheet/json/'

    if os.path.exists(out_of_sample_dir):
        for file in os.listdir(out_of_sample_dir):
            shutil.move(os.path.join(out_of_sample_dir, file), os.path.join(cricksheet_dir + file))

            

def download_and_preprocess():
    
    print("Running execute_scraper()")
    execute_scraper()

    print("Runnning integrate sample data")
    integrate_sample_data()
    
    print("Running json_generator()")
    json_generator()
    
    rename_date()
    
    print("Running mw_overall_generator()")
    mw_overall_generator()

    print("Running aggregate()")
    aggregate()
    
    print("Running adding_names()")
    adding_names()

    print("Running style_based_features()")
    style_based_features()

download_and_preprocess()