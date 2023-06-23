import discord
import requests
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Get NBA game data for 2023 season
nba_api_url = "https://api.sportsdata.io/v3/nba/scores/json/Games/2023?key={API_KEY}"
nba_response = requests.get(nba_api_url)
nba_data = json.loads(nba_response.text)
nba_df = pd.json_normalize(nba_data)

# One-hot encode the categorical features
nba_X = pd.get_dummies(nba_df[['HomeTeam', 'AwayTeam', 'StadiumID']])
# print(nba_X)

# Create binary target variable
nba_y = nba_df['AwayTeamScore'] > nba_df['HomeTeamScore']
# print(nba_y)

# Split data into training and testing sets
nba_train_size = int(len(nba_df) * 0.8)
nba_X_train, nba_X_test = nba_X[:nba_train_size], nba_X[nba_train_size:]
nba_y_train, nba_y_test = nba_y[:nba_train_size], nba_y[nba_train_size:]

# Train random forest model and make predictions
nba_model = RandomForestClassifier()
nba_model.fit(nba_X_train, nba_y_train)
nba_y_pred = nba_model.predict(nba_X_test)
nba_accuracy = accuracy_score(nba_y_test, nba_y_pred)

# Combine test data with predicted results
test_df = nba_df[nba_train_size:].reset_index(drop=True)
test_df['PredictedResult'] = nba_y_pred

# Add date of game to output
test_df['GameDate'] = pd.to_datetime(test_df['DateTime']).dt.date

# Store predictions in a list of dictionaries
predictions = []
for i, row in test_df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    predicted_result = 'Away' if row['PredictedResult'] else 'Home'
    game_date = row['GameDate']
    prediction = {
        'GameDate': str(game_date),
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'PredictedWinner': predicted_result
    }
    predictions.append(prediction)

# Get MLB game data for 2023 season
mlb_api_url = "https://api.sportsdata.io/api/mlb/odds/json/Games/2023?key={API-KEY}"
mlb_response = requests.get(mlb_api_url)
mlb_data = json.loads(mlb_response.text)
mlb_df = pd.json_normalize(mlb_data)

# One-hot encode the categorical features
mlb_X = pd.get_dummies(mlb_df[['HomeTeam', 'AwayTeam', 'StadiumID']])

# Fill missing values with the mean
imputer = SimpleImputer()
mlb_X = imputer.fit_transform(mlb_X)

# Create binary target variable
mlb_y = mlb_df['AwayTeamRuns'] > mlb_df['HomeTeamRuns']

# Fill missing values with the mean
imputer = SimpleImputer(strategy='median')
mlb_y = imputer.fit_transform(mlb_y.to_numpy().reshape(-1, 1)).flatten()


# Split data into training and testing sets
mlb_train_size = int(len(mlb_df) * 0.04)
mlb_X_train, mlb_X_test = mlb_X[:mlb_train_size], mlb_X[mlb_train_size:]
mlb_y_train, mlb_y_test = mlb_y[:mlb_train_size], mlb_y[mlb_train_size:]

# Train random forest model and make predictions
mlb_model = RandomForestClassifier()
mlb_model.fit(mlb_X_train, mlb_y_train)
mlb_y_pred = mlb_model.predict(mlb_X_test)
mlb_accuracy = accuracy_score(mlb_y_test, mlb_y_pred)

# Combine test data with predicted results
test_df = mlb_df[mlb_train_size:].reset_index(drop=True)
test_df['PredictedResult'] = mlb_y_pred

# Add date of game to output
test_df['GameDate'] = pd.to_datetime(test_df['DateTime']).dt.date

# Store predictions in a list of dictionaries
predictions_mlb = []
for i, row in test_df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    predicted_result = 'Away' if row['PredictedResult'] else 'Home'
    game_date = row['GameDate']
    prediction = {
        'GameDate': str(game_date),
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'PredictedWinner': predicted_result
    }
    predictions_mlb.append(prediction)

# print(predictions)

def mlb_predictions(date):
    # print(f"Predictions for {date}:")
    results = []
    for prediction in predictions_mlb:
        if prediction['GameDate'] == date:
            results.append(f"{prediction['AwayTeam']} vs {prediction['HomeTeam']}: {prediction['PredictedWinner']}")
    return results

def nba_predictions(date):
    # print(f"Predictions for {date}:")
    results = []
    for prediction in predictions:
        if prediction['GameDate'] == date:
            results.append(f"{prediction['AwayTeam']} vs {prediction['HomeTeam']}: {prediction['PredictedWinner']}")
    return results

# Get NFL game data for 2023 season
nfl_api_url = "https://api.sportsdata.io/v3/nfl/scores/json/Games/2023?key=1b0d2025a1a1407bb9e0b544986a80d9"
nfl_response = requests.get(nfl_api_url)


def nfl_games(date):
    try:
        nfl_data = json.loads(nfl_response.text)
        nfl_df = pd.json_normalize(nfl_data)
        # One-hot encode the categorical features
        nfl_X = pd.get_dummies(nfl_df[['HomeTeam', 'AwayTeam', 'StadiumID']])
        # Create binary target variable
        nfl_y = nfl_df['AwayScore'] > nfl_df['HomeScore']
        # Split data into training and testing sets
        nfl_train_size = int(len(nfl_df) * 0.08)
        nfl_X_train, nfl_X_test = nfl_X[:nfl_train_size], nfl_X[nfl_train_size:]
        nfl_y_train, nfl_y_test = nfl_y[:nfl_train_size], nfl_y[nfl_train_size:]
        # Train random forest model and make predictions
        nfl_model = RandomForestClassifier()
        nfl_model.fit(nfl_X_train, nfl_y_train)
        nfl_y_pred = nfl_model.predict(nfl_X_test)
        nfl_accuracy = accuracy_score(nfl_y_test, nfl_y_pred)
        # Combine test data with predicted results
        test_df = nfl_df[nfl_train_size:].reset_index(drop=True)
        test_df['PredictedResult'] = nfl_y_pred
        # Add date of game to output
        test_df['GameDate'] = pd.to_datetime(test_df['DateTime']).dt.date
        # Store predictions in a list of dictionaries
        predictions = []
        for i, row in test_df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            predicted_result = 'Away' if row['PredictedResult'] else 'Home'
            game_date = row['GameDate']
            prediction = {
                'GameDate': str(game_date),
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'PredictedWinner': predicted_result
            }
            predictions.append(prediction)
            # Print or do something with predictions

            # print(f"Predictions for {date}:")
            results = []
            for prediction in predictions:
                if prediction['GameDate'] == date:
                    results.append(f"{prediction['AwayTeam']} vs {prediction['HomeTeam']}: {prediction['PredictedWinner']}")
            return results

    except KeyError:
        return (["Required data not available in API response."])


# nfl_games('2023-09-10')
# client = discord.Client()
guild = discord.Guild
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    else:
        if message.content.lower() == "hello" and message.channel.id == {CHANNEL_ID}:
            await message.channel.send(
                "If you are here for game prediction, kindly go to the respective channels to run the commands. If not, Have Fun!\nThank you! ")
        # else:
        #     await message.channel.send(
        #         "How may I be of service today? \n Please enter: \n 'mlb' for MLB's games \n 'nba' for NBA's games")

        if message.content.lower() == "nba" and message.channel.id == {CHANNEL_ID}:
            await message.channel.send("Please, just enter the date for the game in this format: yyyy-mm-dd")
            next_message = await client.wait_for('message')
            await message.channel.send("Please wait, while I process the games. ")
            next_message_content = next_message.content
            output = nba_predictions(next_message_content)
            if len(output) == 0:
                await message.channel.send("No games found for the given date.")
            else:
                await message.channel.send(f"Games Predictions for {next_message_content}:")
                await message.channel.send("\n".join(output))
            # print(output)
            # await message.channel.send(f" \n\n\n\n\n The results for {next_message_content} games are: \n {output}")
        elif message.content.lower() == "mlb" and message.channel.id == {CHANNEL_ID}:
            await message.channel.send("Please, just enter the date for the game in this format: yyyy-mm-dd")
            next_message = await client.wait_for('message')
            await message.channel.send("Please wait, while I process the games. ")
            next_message_content = next_message.content
            output = mlb_predictions(next_message_content)
            if len(output) == 0:
                await message.channel.send("No games found for the given date.")
            else:
                await message.channel.send(f"Games Predictions for {next_message_content}:")
                await message.channel.send("\n".join(output))

        elif message.content.lower() == "nfl" and message.channel.id == {CHANNEL_ID}:
            await message.channel.send("Please, just enter the date for the game in this format: yyyy-mm-dd")
            next_message = await client.wait_for('message')
            await message.channel.send("Please wait, while I process the games. ")
            next_message_content = next_message.content
            output = nfl_games(next_message_content)
            if len(output) == 0:
                await message.channel.send("No games found for the given date.")
            else:
                await message.channel.send(f"Games Predictions for {next_message_content}:")
                await message.channel.send("\n".join(output))
        # else:
        #     await message.channel.send("You are not in the right channel you run")

client.run("{DISCORD_TOKEN}")
