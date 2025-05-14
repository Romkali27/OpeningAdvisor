import berserk
import base64
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import xgboost
import json
import numpy as np
import seaborn as sns
import warnings
import dotenv
import os
import joblib
from sklearn.preprocessing import StandardScaler


def get_data(nickname):
    session = berserk.TokenSession("")
    client = berserk.Client(session=session)
    warnings.filterwarnings("ignore")

    rating_thresholds = {1200: 'beginner', 1500: 'amateur', 1800: 'upper-intermidiate', 2150: 'advanced',
                         100000: 'specialist'}
    type_thresholds = {
        0: {'beginner': [[5, 6.5], 4.75], 'amateur': [[5.5, 7], 5], 'upper-intermidiate': [[6, 7.5], 5.5],
            'advanced': [[5.5, 7.5], 5], 'specialist': [[5, 6.5], 4.75]},
        1: {'beginner': [[0.65, 0.825], 38], 'amateur': [[0.55, 0.7], 41], 'upper-intermidiate': [[0.45, 0.575], 44],
            'advanced': [[0.375, 0.475], 47], 'specialist': [[0.335, 0.425], 50]},
        2: {'beginner': [[0.45, 0.55], 5], 'amateur': [[0.35, 0.45], 6], 'upper-intermidiate': [[0.25, 0.35], 6.5],
            'advanced': [[0.175, 0.275], 7], 'specialist': [[0.125, 0.2], 7.5]},
        3: {'beginner': [[0.75, 1.15], 0.4], 'amateur': [[0.7, 1], 0.35], 'upper-intermidiate': [[0.65, 0.9], 0.3],
            'advanced': [[0.6, 0.8], 0.25], 'specialist': [[0.58, 0.75], 0.22]}
    }

    raw_data = client.games.export_by_player(nickname, max=100, evals=True, clocks=True, opening=True,
                                             perf_type=['blitz', 'rapid'], analysed=True, tags=False)
    ## In the line above, we use the client instance, established using our token to import games of a player.
    ## The parameters of the function in brackets specify the details of what data should be imported.
    ## In this case, the data is 100 most recent analyzed games with evaluations, time usage and opening data from blitz and rapid time control.
    games = list(raw_data)
    ## Notice that the imported data is called raw_data. The reason is that you have to change it's type for it to be analyzable.
    ## We do this by organizing it in a list, which will contain each game in an order - from most to least recent.
    if len(games) < 10:
        print('Not enough analyzed games yet!')
        return None, None, None
    ## Now that the data is readable, we ensure that there is enough material to work with.
    ## To do this, we check the length of the list, which is the amount of games imported.
    ## If there are less than 10 games, we print the according message and stop the function.
    opening_swing_list = []  ## In the lines preceeding the for cycle, we are initializing data containers for us to store the game data in.
    blunder_score_list = []
    no_tp_time_usage_means = []
    no_tp_time_usage_stds = []
    no_tp_eval_swing_stds = []
    no_tp_eval_swing_means = []
    game_cnt = 0
    color = ''
    debut_depth_list = []
    eval_swing_means = []
    eval_swing_stds = []
    normalized_time_usage_means = []
    normalized_time_usage_stds = []
    opening_eval_swings = []
    moves_per_game = []
    moves_btp_list = []
    mean = 0

    for game in games:  ## Now we begin to access each game of the list for the data.
        moves_btp = 0  ## Once again, starting with initializing the data containers for future use.
        no_tp_eval_swing_list = []
        no_tp_time_usage_list = []
        increment = int(game['clock']['increment'])
        time_list = []
        eval_swing_list = []
        time_std = 0
        eval_swing_mean = 0
        eval_swing_std = 0
        plies = len(game['moves'].split(' '))
        moves = plies // 2
        game_cnt += 1
        moves_per_game.append(moves)
        if 'opening' in game.keys():  ## To avoid rare occasions where lichess doesn't recognize the opening, we will append 1 as length of an opening.
            debut_depth_list.append(game['opening']['ply'])
        else:
            debut_depth_list.append(1)

        if 'user' in game['players']['white'].keys() and 'user' in game['players'][
            'black'].keys():  ## Now we identify the player's piece color.
            if game['players']['white']['user']['name'].lower() == nickname.lower():
                color = 'w'
                blunder_points = game['players']['white']['analysis']['inaccuracy'] / 2 + \
                                 game['players']['white']['analysis']['mistake'] * 1.5 + \
                                 game['players']['white']['analysis']['blunder'] * 3
                blunder_score = blunder_points / moves  ## (Blunder score = Inaccuracies * 0.5 + Mistakes * 1.5 + Blunders * 3) / Move amount
                blunder_score_list.append(round(blunder_score,
                                                2))  ## At the same time we can calculate the blunder score, which tells the amount of size of mistake.
                mean += game['players']['white'][
                    'rating']  ## Appending the actual rating of a player to identify his rating stage.
            else:
                color = 'b'
                blunder_points = game['players']['black']['analysis']['inaccuracy'] + \
                                 game['players']['black']['analysis']['mistake'] * 2 + \
                                 game['players']['black']['analysis']['blunder'] * 3
                blunder_score = blunder_points / moves
                blunder_score_list.append(round(blunder_score, 2))
                mean += game['players']['black']['rating']
        else:
            pass

        for i in range(len(game['analysis'])):
            if type(game['analysis'][i].get('eval')) != int:
                game['analysis'][i]['mate'] = 1500
                ## To evaluate the evaluation swing, we have to keep them consistent, so when there is a mate in position,
                ## we evaluate it as a 1500 centipawn advantage or disadvantage to avoid miscalculations.

        evals = [i.get('eval') if type(i.get('eval')) == int else i.get('mate') for i in game['analysis']]
        ## Generate a full evaluation list for the game to analyze the swings.

        if plies > len(evals):
            pass
        else:
            for i in range(2, plies - 2):
                if plies > 0:
                    j = i - 1
                    if color == 'w':
                        if i % 2 == 0:
                            move_time = ((int(game['clocks'][i]) - int(game['clocks'][i + 2])) / 100 + increment)
                            time_list.append(round(move_time, 2))
                            eval_swing_list.append(abs(evals[i] - evals[j]))
                            if 'opening' not in game.keys():
                                if int(game['clocks'][i + 2]) > 0.2 * int(game['clocks'][0]) and i // 2 + 1:
                                    no_tp_eval_swing_list.append((abs(evals[i] - evals[j])))
                                    no_tp_time_usage_list.append(round(move_time, 2))
                                    moves_btp += 1
                            ## Now we start to fill our data containers with statistical data like moves before time trouble amount, time usage and evaluation swings.
                            ## Notice that there are separate 'no_tp' versions of lists, which means 'no time pressure'. For the sake of this project, TT = >20% of time.
                            else:
                                if int(game['clocks'][i + 2]) > 0.2 * int(game['clocks'][0]) and i // 2 + 1 > \
                                        game['opening']['ply'] // 2:
                                    no_tp_eval_swing_list.append((abs(evals[i] - evals[j])))
                                    no_tp_time_usage_list.append(round(move_time, 2))
                                    moves_btp += 1
                    else:
                        if i % 2 == 1:
                            move_time = ((int(game['clocks'][i]) - int(game['clocks'][i + 2])) / 100 + increment)
                            time_list.append(round(move_time, 2))
                            eval_swing_list.append(abs(evals[i] - evals[j]))
                            if 'opening' not in game.keys():
                                if int(game['clocks'][i + 2]) > 0.2 * int(game['clocks'][0]) and i // 2 + 1:
                                    no_tp_eval_swing_list.append((abs(evals[i] - evals[j])))
                                    no_tp_time_usage_list.append(round(move_time, 2))
                                    moves_btp += 1
                            else:
                                if int(game['clocks'][i + 2]) > 0.2 * int(game['clocks'][0]) and i // 2 + 1 > \
                                        game['opening']['ply'] // 2:
                                    no_tp_eval_swing_list.append((abs(evals[i] - evals[j])))
                                    no_tp_time_usage_list.append(round(move_time, 2))
                                    moves_btp += 1

        no_tp_eval_swing_srs = pd.Series(no_tp_eval_swing_list) / 100
        nrm_no_tp_time_usage_srs = pd.Series(no_tp_time_usage_list) / (
                    game['clock']['initial'] + increment * moves) * 100
        opening_swing_list.append(
            round((sum(eval_swing_list[0:10]) / 1000), 2) if moves > 11 else round((sum(eval_swing_list) / 1000), 2))
        nrm_time_list_srs = pd.Series(time_list) / (game['clock']['initial'] + increment * moves) * 100
        eval_swing_srs = pd.Series(eval_swing_list) / 100
        debut_depth_srs = pd.Series(debut_depth_list)
        moves_per_game_srs = pd.Series(moves_per_game)
        no_tp_time_usage_srs = pd.Series(no_tp_time_usage_list)
        moves_btp_list.append(moves_btp)
        ## In the code block above, we convert our Python lists to Pandas objects called Series.
        ## The reason is very simple - we can take statistical measures like mean and std from them in just one method.

        no_tp_eval_swing_means.append(round(no_tp_eval_swing_srs.mean(), 2))
        no_tp_eval_swing_stds.append(round(no_tp_eval_swing_srs.std(), 2))
        eval_swing_means.append(round(eval_swing_srs.mean(), 2))
        eval_swing_stds.append(round(eval_swing_srs.std(), 2))
        normalized_time_usage_means.append(round(nrm_time_list_srs.mean(), 2))
        normalized_time_usage_stds.append(round(nrm_time_list_srs.std(), 2))
        no_tp_time_usage_means.append(round(no_tp_time_usage_srs.mean(), 2))
        no_tp_time_usage_stds.append(round(no_tp_time_usage_srs.std(), 2))

        ## And now we do exactly that to populate our end data containers, which will be the analysis data sources.

    blunder_score_srs = pd.Series(blunder_score_list)
    no_tp_eval_swing_stds_pds = pd.Series(no_tp_eval_swing_stds)
    no_tp_eval_swing_means_pds = pd.Series(no_tp_eval_swing_means)
    opening_swing_list_pds = pd.Series(opening_swing_list)
    debut_depth_list_pds = pd.Series(debut_depth_list)
    eval_swing_means_pds = pd.Series(eval_swing_means)
    eval_swing_stds_pds = pd.Series(eval_swing_stds)
    normalized_time_usage_means_pds = pd.Series(normalized_time_usage_means)
    normalized_time_usage_stds_pds = pd.Series(normalized_time_usage_means)
    moves_per_game_pds = pd.Series(moves_per_game)
    no_tp_time_usage_means_pds = pd.Series(no_tp_time_usage_means)
    no_tp_time_usage_stds_pds = pd.Series(no_tp_time_usage_stds)
    moves_btp_pds = pd.Series(moves_btp_list)
    ## All that's left to do is to convert these end data containers into Series themselves.

    result_type = ''  ## Initializing a text object, which will contain the final rating type.
    mean //= game_cnt
    for key in list(rating_thresholds.keys()):
        if mean < key:
            stage = rating_thresholds[key]
            break
    ## Determine the rating stage by taking the total ratings, dividing them by the amount of games and checking, which stage fits the result.

    type_criterions = {
        0: [no_tp_time_usage_stds_pds.mean(), no_tp_time_usage_means_pds.mean(), 'IU'],
        1: [no_tp_eval_swing_means_pds.mean(), moves_per_game_pds.quantile(q=0.75), 'GH'],
        2: [opening_swing_list_pds.mean(), debut_depth_list_pds.mean(), 'TN'],
        3: [no_tp_eval_swing_stds_pds.mean(), blunder_score_srs.mean(), 'LV']
    }
    ## The dictionary above shows which criteria will be used to conduct the primary analysis and to determine the result MBTI.
    ## For U/I - Time usage STD/Time usage mean
    ## For G/H - Evaluation swing mean, 75% quantile of move amount
    ## For T/N - Evaluation swing in the opening/Length of opening theory mean
    ## For L/V - Evaluation swing STD/Blunder score mean

    for i in range(len(type_thresholds)):
        criteria_min_value = type_thresholds[i][stage][0][0]
        criteria_max_value = type_thresholds[i][stage][0][1]
        criteria_second_value = type_thresholds[i][stage][1]
        user_first_criteria_value = type_criterions[i][0]
        user_second_criteria_value = type_criterions[i][1]
        if i not in [1, 2]:
            if criteria_min_value >= user_first_criteria_value:
                result_type += (type_criterions[i][2][0])
            elif criteria_max_value <= user_first_criteria_value:
                result_type += type_criterions[i][2][1]
            else:
                if criteria_second_value > user_second_criteria_value:
                    result_type += (type_criterions[i][2][0])
                else:
                    result_type += (type_criterions[i][2][1])
        else:
            if criteria_min_value > user_first_criteria_value:
                result_type += (type_criterions[i][2][0])
            elif criteria_max_value < user_first_criteria_value:
                result_type += type_criterions[i][2][1]
            else:
                if criteria_second_value > user_second_criteria_value:
                    result_type += (type_criterions[i][2][1])
                else:
                    result_type += (type_criterions[i][2][0])
        ## Now we enter the type calculation process. The initial techinque is as follows:
        ## Check the first criteria user value and compare it with the smaller and the larger threshold.
        ## If the value is less than the smaller threshold or larger than the bigger one, the letter is assigned.
        ## If the value is between them, compare the second value with the threshold and assign the letter.

    data = blunder_score_srs.mean(), opening_swing_list_pds.mean(), no_tp_eval_swing_stds_pds.mean(), no_tp_eval_swing_means_pds.mean(), debut_depth_list_pds.mean(), eval_swing_means_pds.mean(), eval_swing_stds_pds.mean(), normalized_time_usage_means_pds.mean(), normalized_time_usage_stds_pds.mean(), moves_per_game_pds.quantile(
        q=0.75), no_tp_time_usage_means_pds.mean(), no_tp_time_usage_stds_pds.mean(), moves_btp_pds.mean()
    X = pd.DataFrame(data=[data],
                     columns=['Blunder scores', 'Eval swings (first 10 moves)', 'Eval swings std (middlegame)',
                              'Eval swings mean (middlegame)', 'Opening depth', 'Mean eval swings', 'Eval swing std',
                              'Mean time usage per move (%)', 'Time usage std per move (%)', 'Moves per game',
                              'Mean time usage per move (%, middlegame)', 'Time usage std per move (%, middlegame)',
                              'Moves before time pressure'])

    type_ = ''
    M1 = joblib.load('M1.pkl')
    M2 = joblib.load('M2.pkl')
    M3 = joblib.load('M3.pkl')
    M4 = joblib.load('M4.pkl')

    data_df = pd.read_csv('df1.csv')
    data_df.dropna(inplace=True, axis=0)
    Scaler = StandardScaler()
    Scaler.fit(data_df)
    Scaler.transform(data_df)
    X = Scaler.transform(X)

    type_ += M1.predict(X)[0]
    type_ += M2.predict(X)[0]
    type_ += M3.predict(X)[0]
    type_ += M4.predict(X)[0]
    return X, type_, stage


