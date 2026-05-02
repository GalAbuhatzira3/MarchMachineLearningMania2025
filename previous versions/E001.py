from __future__ import annotations
from pathlib import Path

# ------------------------------------------------- Imports ---------------------------------------------------------- #
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# ------------------------------------------------- Paths + file names ----------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Auto-detect project paths
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "submission.csv"  # output file name and path

# ------------------------------------------------- Specific CSVs ---------------------------------------------------- #
# regular season info:
MEN_REG = "MRegularSeasonCompactResults.csv"
WOMEN_REG = "WRegularSeasonCompactResults.csv"
DAYS_IN_REGULAR_SEASON = 132

# tournament results:
MEN_TOUR = "MNCAATourneyCompactResults.csv"
WOMEN_TOUR = "WNCAATourneyCompactResults.csv"

# the matchups we must predict:
SAMPLE_SUB = "SampleSubmissionStage2.csv"


# ------------------------------ CSV Handling ------------------------------------------------------------------------ #
def read_csv(name: str, usecols=None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, usecols=usecols)  # usecols loads only columns we need (faster + less memory)


# ---------------------------- Building (TeamID, Season) Stats Table ------------------------------------------------- #
def build_team_season_features(regular: pd.DataFrame) -> pd.DataFrame:
    # Entering here, 'regular' is the concatenated table of the regular season data of men and women.
    # What we want from it is to return a table that extracts stats for each team in each season.
    # Winner perspective:
    winner = regular[["Season", "WTeamID", "WScore", "LScore"]].copy()
    winner.columns = ["Season", "TeamID", "PF", "PA"]
    winner["G"] = 1  # this was one game
    winner["W"] = 1  # this game was a win
    # Loser perspective:
    loser = regular[["Season", "LTeamID", "LScore", "WScore"]].copy()
    loser.columns = ["Season", "TeamID", "PF", "PA"]
    loser["G"] = 1  # this was one game
    loser["W"] = 0  # this game wasn't a win
    # Concatenation of winners and losers:
    games = pd.concat([winner, loser], ignore_index=True)
    # Building the feats table as grouping over the key (Season, TeamID) because we want the stats of the team
    # for every team and every season
    feats = games.groupby(["Season", "TeamID"], as_index=False).agg(
        G=("G", "sum"),  # summing gives us TOTAL games TeamID played in Season
        W=("W", "sum"),  # summing gives us TOTAL wins TeamID won in Season
        PF=("PF", "sum"),  # summing gives us TOTAL points TeamID scored in Season
        PA=("PA", "sum"),  # summing gives us TOTAL points TeamID conceded in Season
    )
    # Convert totals into rates:
    feats["WinPct"] = feats["W"] / feats["G"]  # winning percentage
    feats["PF_perG"] = feats["PF"] / feats["G"]  # points scored per game
    feats["PA_perG"] = feats["PA"] / feats["G"]  # points conceded per game
    feats["Diff_perG"] = (feats["PF"] - feats["PA"]) / feats["G"]  # scoring margin per game
    # Return only the relevant columns: (without the ones used for the calculations)
    return feats[["Season", "TeamID", "WinPct", "PF_perG", "PA_perG", "Diff_perG", "G"]]


# ---------------------------- Building supervised learning dataset (X,y) -------------------------------------------- #
def build_training_from_tournament(tournament: pd.DataFrame, feats: pd.DataFrame):
    # Entering here, 'tournament' is the table containing the data of tournaments (men + women)
    # and 'feats' is the stats table we've built from the regular season data (men + women)
    # Now, one tournament game becomes one training row:
    #   input features: something about the teams A and B where A.id < B.id by convention
    #   label: if(A wins) y = 1; else y = 0

    # df columns = [Season, WTeamID, LTeamID] (row example: [1985, 1116, 1234])
    df = tournament.copy()
    # Now we want to add attribute A to be the TeamID with the lower ID between the two, and B the other one:
    df["A"] = df[["WTeamID", "LTeamID"]].min(axis=1)  # in that case A = 1116
    df["B"] = df[["WTeamID", "LTeamID"]].max(axis=1)  # in that case B = 1234
    # and if A is the winner, the attribute y will be 1 otherwise 0:
    df["y"] = (df["A"] == df["WTeamID"]).astype(int)  # in that case WTeamID = 1116 so y = 1

    # Now, take feats and create two copies of it. To the first copy, add A_(attribute) to each attribute other than
    # 'Season' and 'TeamID'. To the other copy, add B_(attribute) in the same way.
    featsA = feats.rename(columns={c: f"A_{c}" for c in feats.columns if c not in ["Season", "TeamID"]})
    featsB = feats.rename(columns={c: f"B_{c}" for c in feats.columns if c not in ["Season", "TeamID"]})

    # Now, merge (just a JOIN) features into each game row:
    # attach A_... features by matching (Season, A)
    merged = df.merge(featsA, left_on=["Season", "A"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])
    # attach B_... features by matching (Season, B)
    merged = merged.merge(featsB, left_on=["Season", "B"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])
    # Error handling: if a team-season feature is missing for some reason, we use 0.0
    merged = merged.fillna(0.0)
    # Now, after the last merge, each row in 'merged' corresponds to a tournament game between teams A and B and
    # also include the summaries of A's season and B's season, and who won the tournament game.

    # Now, from 'merged' we build the actual model input X = (A - B).
    # We're giving the model the differences between each 2 corresponding attributes (A_... - B_...) in a row,
    # so he will determine the chances of A to win the match.
    # If A is stronger than B, then the differences should be all positive.
    base = ["WinPct", "PF_perG", "PA_perG", "Diff_perG", "G"]
    X = pd.DataFrame({k: merged[f"A_{k}"] - merged[f"B_{k}"] for k in base})

    # y is a list of 1s and 0s. y[i] = 1 <-> A won the tournament game that the i'th row in 'merged' represents.
    y = merged["y"].to_numpy()
    groups = merged["Season"].to_numpy()
    return X, y, groups


def parse_submission_ids(sub: pd.DataFrame):
    parts = sub["ID"].str.split("_", expand=True)
    season = parts[0].astype(int)
    a = parts[1].astype(int)
    b = parts[2].astype(int)
    return season, a, b


def build_matchup_features(season: pd.Series, a: pd.Series, b: pd.Series, feats: pd.DataFrame) -> pd.DataFrame:
    keysA = pd.DataFrame({"Season": season, "TeamID": a})
    keysB = pd.DataFrame({"Season": season, "TeamID": b})

    fA = keysA.merge(feats, on=["Season", "TeamID"], how="left").fillna(0.0)
    fB = keysB.merge(feats, on=["Season", "TeamID"], how="left").fillna(0.0)

    base = ["WinPct", "PF_perG", "PA_perG", "Diff_perG", "G"]
    X = pd.DataFrame({k: fA[k] - fB[k] for k in base})
    return X


def get_regular():
    def extract_regular_from_csv(csv):
        # Each row represents one game and has the columns:
        regular_columns = ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore"]
        regular_data = read_csv(csv, usecols=regular_columns)
        regular_data = regular_data[regular_data["DayNum"] <= DAYS_IN_REGULAR_SEASON]
        return regular_data

    men_regular = extract_regular_from_csv(MEN_REG)  # men regular season results
    women_regular = extract_regular_from_csv(WOMEN_REG)  # women regular season results
    # Concatenate to one table:
    regular = pd.concat([men_regular, women_regular], ignore_index=True)
    print(f"Regular season loaded! Loaded games: {len(regular):,}")
    return regular


def get_tournament():
    # Each row is a real tournament game, and has the columns:
    tournament_columns = ["Season", "WTeamID", "LTeamID"]
    men_tournament = read_csv(MEN_TOUR, usecols=tournament_columns)  # men regular season results
    women_tournament = read_csv(WOMEN_TOUR, usecols=tournament_columns)  # women regular season results
    # Concatenate to one table:
    tournament = pd.concat([men_tournament, women_tournament], ignore_index=True)
    print(f"Tournament loaded! Loaded games: {len(tournament):,}")
    return tournament


# ------------------------------------------------- Main ------------------------------------------------------------- #
def main():
    # we want a function that, for any match-up ID like 2025_1101_1104, outputs:
    #   pred = P[team 1101 beats team 1104 in season 2025]
    # to get that, we use history: past seasons where we know who actually won.
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir:      {DATA_DIR}")

    # --------------------------------- Loading Data ----------------------------------------------------------------- #
    # For this initial version we load 3 types of data (history):
    # (A) Regular season games (men + women)
    #   Purpose: This will be used to describe teams (their strength) during the season.
    regular = get_regular()

    # (B) Tournament games (men + women)
    #   Purpose: This will be our training set (labels) - who actually won in the tournament
    tournament = get_tournament()

    # (C) The matchups we must predict
    # SampleSubmissionStage2.csv (131,407 rows) contains ID like 2025_1101_1102,  2025_1101_1103
    #   Purpose: this is the list of all matchups Kaggle expects in our submission.
    #            We must output a probability for each.
    sub = read_csv(SAMPLE_SUB, usecols=["ID", "Pred"])
    print(f"Loaded submission rows: {len(sub):,}")

    # --------------------------------- Training --------------------------------------------------------------------- #
    # [1] Turn regular-season games into team season stats:
    # we simply want to extract information about the season a given team had (in terms of winning pct, PF per game ...)
    # and we want to do it for every team (by TeamID) and for every season.
    # It's the model's input vocabulary: a team is not simply "1101" anymore, it's now a team that wins 75% and has +7.9
    # points differential per game.
    # Theory-wise: we are making the assumption that tournament strength is correlated with regular-season performance,
    # as better teams win more games, score more, concede less (again, just an initial version)
    # input: regular table where EACH ROW IS A GAME, in a certain season, and it has a winner and a loser
    # and their score
    # output: a table where each row is a summary of a season a certain team had, in terms of winning percentage,
    # points scored average, points conceded average, average differential and total games played.
    features = build_team_season_features(regular)

    # [2] Build a supervised learning dataset (X,y)
    # Each row in X will correspond to one tournament game between TeamA and TeamB.
    # Each such row, will include the difference between regular season stats of A and B (i.e. A - B).
    # The intuition for it, is that if A is considered stronger than B, then those differences would likely to be
    # positive (better win% for A, better PPG for A and such).
    # the list y will contain the results of those matchups, from the training set.
    # y[i] = 1 <-> A won the tournament game that the i'th row in 'merged' represents.
    X, y, groups = build_training_from_tournament(tournament, features)

    print(f"Feature rows (team-season): {len(features):,}")
    print(f"Training rows: {len(X):,} | Positive rate: {y.mean():.3f}")

    # --------------------------------- Building the Model ----------------------------------------------------------- #
    # The 1st model will be based on logistic regression. It learns the weighted sum of out feature diffs, Then turns
    # it into a probability p so:
    #   if A looks much stronger than B -> p ~ 1
    #   if A looks much weaker than B -> p ~ 0
    #   if similar -> p ~ 0.5
    # The model fits Kaggle perfectly, because it wants probabilities and not just binary predictions of 1 or 0.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])

    # --------------------------------- Validating with season-based CV (brier score) -------------------------------- #
    # What we want to achieve in this section is to see how good our model is. Meaning, how well the training goes:
    # given a dataset X with known results y, we want to see how close y_hat (list), obtained from the model on X,
    # to the real results y.
    # We split by season because we want to simulate predicting a future season from past seasons.
    folds = 5
    gkf = GroupKFold(n_splits=folds)
    briers = []
    # brier score is the mean squared error on probabilities. Brier =  (y_hat - y)^2 (below 0.25 is ok, 0.01 great)
    for i, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]  # [P[A loses], P[A wins]] , we need P[A wins]
        b = brier_score_loss(y[va], p)
        briers.append(b)
        print(f"Fold {i}/{folds}: Brier={b:.5f}")
    # Summarizing CV results:
    print(f"CV Brier mean={np.mean(briers):.5f} | std={np.std(briers):.5f}")

    # --------------------------------- Train on all history --------------------------------------------------------- #
    # After CV we now using every tournament game to learn the best weights:
    model.fit(X, y)

    # --------------------------------- Submission features (Actual prediction) -------------------------------------- #
    # Here we actually want to predict the outcome of the games we don't know the results to. Kaggle gave us a list of
    # matchups, and only he has the correct labels for them (who won each matchup), and we need to give our model the
    # dataset, and he will predict the outcomes.
    # Extract Season, A, B for each row in the submission file:
    season, a, b = parse_submission_ids(sub)
    # looks up feats for (Season, A) and (Season, B) and creates the same diff features as training:
    X_sub = build_matchup_features(season, a, b, features)

    # predict probabilities + clip: (producing the probability that the lower TeamID wins)
    pred = model.predict_proba(X_sub)[:, 1]
    pred = np.clip(pred, 1e-5, 1 - 1e-5)

    # write submission.csv
    out = pd.DataFrame({"ID": sub["ID"], "Pred": pred})
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    # idea:
    # convert regular season results into team-season summaries (feats)
    # convert tournaments games into training rows: X = feats(A) - feats(B), y = (did A win?)
    # use season-based CV to estimate generalization (Brier)
    # train on all history
    # predict all matchups in Stage2 and write a submission file
    main()
