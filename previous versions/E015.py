# ------------------------------------------------- Imports ---------------------------------------------------------- #
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.interpolate import UnivariateSpline

import xgboost as xgb

param = {}
param["objective"] = "reg:squarederror"
param["booster"] = "gbtree"
param["eta"] = 0.01
param["subsample"] = 0.6
param["colsample_bynode"] = 0.8
param["num_parallel_tree"] = 2
param["min_child_weight"] = 4
param["max_depth"] = 4
param["tree_method"] = "hist"
param['grow_policy'] = 'lossguide'
param["max_bin"] = 32

num_rounds = 700

# ------------------------------------------------- Settings --------------------------------------------------------- #
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ------------------------------------------------- Paths + file names ----------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "submission.csv"

# ------------------------------------------------- CSVs ------------------------------------------------------------- #
MEN_REG = "MRegularSeasonCompactResults.csv"
WOMEN_REG = "WRegularSeasonCompactResults.csv"

MEN_REG_DET = "MRegularSeasonDetailedResults.csv"
WOMEN_REG_DET = "WRegularSeasonDetailedResults.csv"

MEN_TOUR = "MNCAATourneyCompactResults.csv"
WOMEN_TOUR = "WNCAATourneyCompactResults.csv"

MEN_TOUR_DET = "MNCAATourneyDetailedResults.csv"
WOMEN_TOUR_DET = "WNCAATourneyDetailedResults.csv"

MEN_SEEDS = "MNCAATourneySeeds.csv"
WOMEN_SEEDS = "WNCAATourneySeeds.csv"

MEN_CONF = "MTeamConferences.csv"
WOMEN_CONF = "WTeamConferences.csv"

MASSEY_ORDINALS = "MMasseyOrdinals.csv"
SAMPLE_SUB = "SampleSubmissionStage2.csv"

# ------------------------------------------------- Constants -------------------------------------------------------- #
REG_FINAL_DAY = 132


# ------------------------------ Loading Data ------------------------------------------------------------------------ #
def adjust_overtime(df, is_detailed=False):
    """adjust game statistics due to overtime"""
    com_columns = ["WScore", "LScore"]
    det_columns = ["LScore", "WScore",
                   "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
                   "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]
    for col in (det_columns if is_detailed else com_columns):
        df[col] = df[col] / ((40 + 5 * df["NumOT"]) / 40)
    return df.drop(columns=["NumOT"])  # after that we have no need for NumOT


def read_csv(name: str, usecols=None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, usecols=usecols)


def get_conference():
    men_team_conf = read_csv(MEN_CONF)
    women_team_conf = read_csv(WOMEN_CONF)
    team_confs = pd.concat([men_team_conf, women_team_conf], ignore_index=True)
    return team_confs


def get_tournament(is_detailed: bool):
    men_tournament = read_csv(MEN_TOUR_DET if is_detailed else MEN_TOUR)
    women_tournament = read_csv(WOMEN_TOUR_DET if is_detailed else WOMEN_TOUR)
    tournament = pd.concat([men_tournament, women_tournament], ignore_index=True)
    tournament = adjust_overtime(tournament, is_detailed)
    return tournament


def get_regular(is_detailed: bool):
    men_regular = read_csv(MEN_REG_DET if is_detailed else MEN_REG)
    women_regular = read_csv(WOMEN_REG_DET if is_detailed else WOMEN_REG)
    regular = pd.concat([men_regular, women_regular], ignore_index=True)
    return adjust_overtime(regular, is_detailed)  # compact: 329,928 rows, detailed: 200,590 rows


def get_seeds() -> pd.DataFrame:
    men = read_csv(MEN_SEEDS)  # 1985 - 2025
    women = read_csv(WOMEN_SEEDS)  # 1998 - 2025
    seeds = pd.concat([men, women], ignore_index=True)
    seeds["Seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
    return seeds


def get_massey() -> pd.DataFrame:
    massey = read_csv(MASSEY_ORDINALS)
    massey = massey[massey["RankingDayNum"] == REG_FINAL_DAY + 1].copy()
    massey = massey[massey["SystemName"].isin(["POM"])]
    consensus = massey.groupby(["Season", "TeamID"], as_index=False).agg(MedianRank=("OrdinalRank", "median"))
    # Custom Log Curve
    consensus["RankLog"] = 100 - 4 * np.log(consensus["MedianRank"] + 1) - consensus["MedianRank"] / 22
    return consensus[["Season", "TeamID", "RankLog"]]


# ---------------------------- Build Features ------------------------------------------------------------------------ #

def build_elo(
        regular,
        K=20,
        base=1500,
        new_team_base=1300,
        reversion=0.30,
        hca=75,
        early_season_games=20,
        early_k_boost=0.5
) -> pd.DataFrame:
    games = regular.copy()
    games = games.sort_values(["Season", "DayNum"], kind="mergesort")
    team_confs = get_conference()
    # Conference map: (Season, TeamID) -> ConfAbbrev
    conf_map = {(int(s), int(t)): c for s, t, c in
                team_confs[["Season", "TeamID", "ConfAbbrev"]].itertuples(index=False)}

    def expected_win_prop(rating_A, rating_B) -> float:
        """The expected score / win probability of Team A (rating_A) against Team B (rating_B) """
        return 1.0 / (1 + math.pow(10, (rating_B - rating_A) / 400.0))

    def k_multiplier(games_played):
        # 1.5*K at game 0 -> 1*K at game 20
        g = min(games_played, early_season_games)
        return (1.0 + early_k_boost) - early_k_boost * (g / early_season_games)

    rows = []
    seasons = games["Season"].unique()
    prev_elo = {}  # (TeamID, rating) dict of last season
    for season in seasons:  # for every season 1985 until 2024:
        season_games = games[games["Season"] == season]  # take just the games of this season
        active_teams = set(season_games["WTeamID"]).union(
            set(season_games["LTeamID"]))  # only the teams that played this season
        ratings = {}  # (TeamID, rating) dict of current year
        if not prev_elo:  # this is the first season (1985):
            for teamID in active_teams:  # basic elo rating since this is the 1st season
                ratings[teamID] = base
        else:  # not the first season, we have data from previous season:
            # the Elo score of the teams this season, will be the mean of the elo score of the teams in their conference
            # last season:
            score = {}
            cnt = {}
            for teamID, rating in prev_elo.items():  # for every team and their rating from last season:
                conf = conf_map.get((int(season) - 1, teamID))
                score[conf] = score.get(conf, 0) + rating
                cnt[conf] = cnt.get(conf, 0) + 1
            prev_conf_mean_elo = {c: score[c] / cnt[c] for c in score if cnt[c] > 0}
            # APPLYING: Year-to-year carryover / mean reversion:
            for teamID in active_teams:  # for every team that participates this season:
                if teamID in prev_elo:  # the team also played last year:
                    conf = conf_map.get((int(season) - 1, teamID))
                    mu = prev_conf_mean_elo.get(conf, base) if conf is not None else base
                    ratings[teamID] = (1 - reversion) * prev_elo[teamID] + reversion * mu
                else:  # new-to-our-history team: start lower
                    ratings[teamID] = new_team_base
        # Track “games played so far this season” for early-season K schedule
        game_played = {t: 0 for t in active_teams}
        for Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc in season_games.itertuples(index=False):
            # APPLYING: Home-court advantage term:
            winner_rating_adj = ratings.get(WTeamID) + (hca if (WLoc == "H") else 0)  # The winner won at home.
            loser_rating_adj = ratings.get(LTeamID) + (hca if (WLoc == "A") else 0)  # the winner won at the road.

            expected_winner = expected_win_prop(winner_rating_adj, loser_rating_adj)  # probability of the winner to win

            # APPLYING: Margin-of-Victory (MOV) multiplier:
            mov_multiplier = np.log(WScore - LScore) + 1.0

            # Early-season K: use average of the two teams' multipliers
            k_eff = 0.5 * K * (k_multiplier(game_played[WTeamID]) + k_multiplier(game_played[LTeamID]))

            # Reevaluating the ratings of both winner and loser based on this single game:
            delta = k_eff * (1.0 - expected_winner) * mov_multiplier  # winner S=1
            ratings[WTeamID] += delta
            ratings[LTeamID] -= delta  # preserve total points exactly

            game_played[WTeamID] += 1
            game_played[LTeamID] += 1
        # Save end-of-season
        for teamID in active_teams:
            rows.append((int(season), teamID, float(ratings[teamID])))
        prev_elo = {int(t): float(ratings[int(t)]) for t in active_teams}

    elo = pd.DataFrame(rows, columns=["Season", "TeamID", "EloEnd"])
    elo = elo.sort_values(["Season", "TeamID"]).drop_duplicates(["Season", "TeamID"], keep="last")

    return elo


def build_team_season_compact_features(regular: pd.DataFrame) -> pd.DataFrame:
    winner = regular[["Season", "WTeamID", "WScore", "LScore"]].copy()
    winner.columns = ["Season", "TeamID", "Score", "OppScore"]
    winner["W"] = 1
    loser = regular[["Season", "LTeamID", "LScore", "WScore"]].copy()
    loser.columns = ["Season", "TeamID", "Score", "OppScore"]
    loser["W"] = 0
    games = pd.concat([winner, loser], ignore_index=True)
    feats = games.groupby(["Season", "TeamID"], as_index=False).agg(
        W=("W", "mean"),
        Score=("Score", "mean"),
        OppScore=("OppScore", "mean"), )
    feats["Diff"] = feats["Score"] - feats["OppScore"]
    return feats


def eliminate_winner_loser(df, name1, name2):
    # extract winner as A, loser as B:
    winner = df.copy()
    winner = winner.rename(
        columns=lambda c: (name1 + c[1:]) if c.startswith("W") else ((name2 + c[1:]) if c.startswith("L") else c))
    winner["W"] = 1
    loser = df.copy()
    # extract loser as A, winner as B:
    loser = loser.rename(
        columns=lambda c: (name2 + c[1:]) if c.startswith("W") else ((name1 + c[1:]) if c.startswith("L") else c))
    loser["W"] = 0
    loser[name1 + "Loc"] = loser[name2 + "Loc"]
    loser.drop(columns=[name2 + "Loc"])
    # concatenate and return the result:
    return pd.concat([winner, loser], ignore_index=True)


def build_team_season_detailed_features(regular_detailed: pd.DataFrame) -> pd.DataFrame:
    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return (num / den).fillna(0.0)

    def get_possession(FGA, OR, TO, FTA):
        return (FGA + TO + 0.44 * FTA) - OR

    df = regular_detailed.copy().drop(columns=["DayNum"])
    df["WPoss"] = get_possession(df["WFGA"], df["WOR"], df["WTO"], df["WFTA"])
    df["LPoss"] = get_possession(df["LFGA"], df["LOR"], df["LTO"], df["LFTA"])
    games = eliminate_winner_loser(df, name1="", name2="Opp")
    games = games.drop(columns=["OppTeamID"])
    games["G"] = 1
    # build features:
    averages = games.groupby(["Season", "TeamID"], as_index=False).mean(numeric_only=True)
    averages["Diff"] = averages["Score"] - averages["OppScore"]
    sums = games.groupby(["Season", "TeamID"], as_index=False).sum(numeric_only=True)
    sums["OffEff"] = safe_div(sums["Score"], sums["Poss"])
    sums["DefEff"] = safe_div(sums["OppScore"], sums["OppPoss"])
    sums["NetEff"] = sums["OffEff"] - sums["DefEff"]
    sums["Pace"] = safe_div(sums["Poss"], sums["G"])

    sums["eFG"] = safe_div(sums["FGM"] + 0.5 * sums["FGM3"], sums["FGA"])
    sums["TOV"] = safe_div(sums["TO"], sums["Poss"])
    sums["ORB"] = safe_div(sums["OR"], sums["OR"] + sums["OppDR"])
    sums["FTR"] = safe_div(sums["FTA"], sums["FGA"])

    sums["3PAr"] = safe_div(sums["FGA3"], sums["FGA"])
    sums["3P%"] = safe_div(sums["FGM3"], sums["FGA3"])
    sums["FT%"] = safe_div(sums["FTM"], sums["FTA"])
    sums["TS"] = safe_div(sums["Score"], 2.0 * (sums["FGA"] + 0.44 * sums["FTA"]))

    sums["AST_Rate"] = safe_div(sums["Ast"], sums["Poss"])
    sums["STL_Rate"] = safe_div(sums["Stl"], sums["OppPoss"])
    sums["BLK_Rate"] = safe_div(sums["Blk"], sums["OppFGA"])
    sums["PF_Rate"] = safe_div(sums["PF"], sums["Poss"])

    sums["Def_eFG"] = safe_div(sums["OppFGM"] + 0.5 * sums["OppFGM3"], sums["OppFGA"])
    sums["Def_3PAr"] = safe_div(sums["OppFGA3"], sums["OppFGA"])
    sums["Def_3P%"] = safe_div(sums["OppFGM3"], sums["OppFGA3"])
    sums["Def_FTR"] = safe_div(sums["OppFTA"], sums["OppFGA"])
    sums = sums[["Season", "TeamID",
                 "OffEff", "DefEff", "NetEff", "Pace", "eFG", "TOV", "ORB", "FTR", "3PAr", "3P%", "FT%", "TS",
                 "AST_Rate", "STL_Rate", "BLK_Rate", "PF_Rate", "Def_eFG", "Def_3PAr", "Def_3P%", "Def_FTR"]]
    averages = averages[["Season", "TeamID", "W", "Diff"]]
    return sums.merge(averages, on=["Season", "TeamID"], how="inner")


# ---------------------------- Building supervised learning dataset (X,y) -------------------------------------------- #

def add_features(df, features, is_training):
    features_A = features.rename(columns={c: f"A_{c}" for c in features.columns if c not in ["Season"]})
    features_B = features.rename(columns={c: f"B_{c}" for c in features.columns if c not in ["Season"]})
    df = df.merge(features_A, on=["Season", "A_TeamID"], how="left")
    df = df.merge(features_B, on=["Season", "B_TeamID"], how="left")
    # adding extra features:
    df["IsMen"] = (df["A_TeamID"] < 3000).astype(int)
    # features["SeedDiff"] = features["A_Seed"] - features["B_Seed"]
    # features["EloDiff"] = features["A_EloEnd"] - features["B_EloEnd"]
    return df


def get_training_set(tournament: pd.DataFrame, features: pd.DataFrame):
    tour_games = eliminate_winner_loser(tournament, name1="A_", name2="B_")
    tour_games = tour_games[["Season", "A_TeamID", "A_Score", "B_TeamID", "B_Score"]]
    tour_games["TourDiff"] = tour_games["A_Score"] - tour_games["B_Score"]
    tour_games = tour_games.drop(columns=["A_Score", "B_Score"])
    # distinguish features:
    tour_games = add_features(tour_games, features, is_training=True)
    tour_games = tour_games.drop(columns=["A_TeamID", "B_TeamID"])  # we have no need for those columns when training
    return tour_games


def parse_submission_ids(sub: pd.DataFrame):
    parts = sub["ID"].str.split("_", expand=True)
    season = parts[0].astype(int)
    team_A = parts[1].astype(int)
    team_b = parts[2].astype(int)
    return season, team_A, team_b


# ------------------------------------------------- Main ------------------------------------------------------------- #
def main():
    # --------------------------------- Loading Data ----------------------------------------------------------------- #
    regular_compact = get_regular(is_detailed=False)
    elo = build_elo(regular_compact)  # 1985 - 2025
    seeds = get_seeds()  # 1985 - 2025
    regular_detailed = get_regular(is_detailed=True)
    tournament_detailed = get_tournament(is_detailed=True)

    # --------------------------------- Feature Building ------------------------------------------------------------- #
    # Detailed Matrix
    features_detailed = build_team_season_detailed_features(regular_detailed)
    features_detailed = features_detailed.merge(elo, on=["Season", "TeamID"], how="left")
    features_detailed = features_detailed.merge(seeds, on=["Season", "TeamID"], how="left")
    features_detailed["Seed"] = features_detailed["Seed"].fillna(17)
    # features_detailed["RankLog"] = features_detailed["RankLog"].fillna(100 - 4 * (175.0 + 1) - 175.0 / 22)

    # --------------------------------- Build Supervised Datasets ---------------------------------------------------- #
    tour_games = get_training_set(tournament_detailed, features_detailed)
    # --------------------------------- Building the Models ---------------------------------------------------------- #
    models = {}
    oof_mae = []
    oof_preds = []
    oof_targets = []
    oof_ss = []

    # leave-one-season out models
    print(tour_games)
    for excluded_season in set(tour_games.Season):
        X_train = tour_games[tour_games.Season != excluded_season].drop(columns=["TourDiff", "Season"]).values
        y_train = tour_games[tour_games.Season != excluded_season].drop(columns=["Season"]).TourDiff.values
        X_validation = tour_games[tour_games.Season == excluded_season].drop(columns=["TourDiff", "Season"]).values
        y_validation = tour_games[tour_games.Season == excluded_season].drop(columns=["Season"]).TourDiff.values
        s_validation = tour_games[tour_games.Season == excluded_season].Season.values

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_validation = xgb.DMatrix(X_validation, label=y_validation)
        models[excluded_season] = xgb.train(
            params=param,
            dtrain=d_train,
            num_boost_round=num_rounds,
        )
        preds = models[excluded_season].predict(d_validation)
        print(f"Excluded season: {excluded_season} Mean absolute error: {mean_absolute_error(y_validation, preds)}")
        oof_mae.append(mean_absolute_error(y_validation, preds))
        oof_preds += list(preds)
        oof_targets += list(y_validation)
        oof_ss += list(s_validation)

    print(f"Average mean absolute error: {np.mean(oof_mae)}")

    # --- The Spread-to-Prob Fix (Platt Scaling) ---
    # We use Logistic Regression to map the predicted margin of victory to a smooth probability
    calibrator = LogisticRegression()

    # Reshape predictions into a 2D column for sklearn
    oof_preds_2d = np.array(oof_preds).reshape(-1, 1)
    labels_1d = (np.array(oof_targets) > 0).astype(int)

    calibrator.fit(oof_preds_2d, labels_1d)

    # Calculate the Brier score of our calibrated XGBoost predictions
    xgb_probs = calibrator.predict_proba(oof_preds_2d)[:, 1]
    print(f"Calibrated XGBoost Brier Score: {brier_score_loss(labels_1d, xgb_probs):.5f}")

    # prepare the submission csv:
    sub = read_csv(SAMPLE_SUB, usecols=["ID", "Pred"])
    season, team_A, team_B = parse_submission_ids(sub)
    sub["Season"] = season
    sub["A_TeamID"] = team_A
    sub["B_TeamID"] = team_B
    sub = add_features(sub, features_detailed, is_training=False)

    # run models on given dataset
    preds = []
    for excluded_season in set(tour_games.Season):
        dtest = xgb.DMatrix(sub.drop(columns=["ID", "Pred", "Season", "A_TeamID", "B_TeamID"]))

        # Predict the Margin of Victory
        margin_preds = models[excluded_season].predict(dtest)

        # Convert Margin to Probability using our Calibrator
        probs = calibrator.predict_proba(margin_preds.reshape(-1, 1))[:, 1]
        preds.append(probs)

    sub['Pred'] = np.array(preds).mean(axis=0)

    out = pd.DataFrame({"ID": sub["ID"], "Pred": sub['Pred']})
    out.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
