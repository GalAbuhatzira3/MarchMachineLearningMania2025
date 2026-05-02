# ------------------------------------------------- Imports ---------------------------------------------------------- #
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

def neg_brier_scorer(estimator, X, y):
    p = estimator.predict_proba(X)[:, 1]
    return -brier_score_loss(y, p)   # higher is better


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


def rank_features_by_brier_perm_importance(
    train_df: pd.DataFrame,
    target_col: str = "TourDiff",
    group_col: str = "Season",
    drop_cols=("Season",),
    n_splits: int = 10,
    n_repeats: int = 5,
    random_state: int = 42,
):
    # Binary label for Kaggle-style probability (win=1)
    y = (train_df[target_col].values > 0).astype(int)
    groups = train_df[group_col].values

    X = train_df.drop(columns=[target_col], errors="ignore").copy()
    X = X.drop(columns=list(drop_cols), errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)

    scorer = neg_brier_scorer

    gkf = GroupKFold(n_splits=n_splits)

    briers = []
    imps = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=1200,
            learning_rate=0.01,
            max_depth=4,
            min_child_weight=4,
            subsample=0.6,
            colsample_bynode=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state + fold,
        )

        model.fit(X.iloc[tr], y[tr])

        p = model.predict_proba(X.iloc[va])[:, 1]
        briers.append(brier_score_loss(y[va], p))

        r = permutation_importance(
            model,
            X.iloc[va],
            y[va],
            scoring=scorer,          # negative Brier
            n_repeats=n_repeats,
            random_state=random_state + fold,
            n_jobs=-1,
        )
        # For negative Brier scorer: importance_mean ≈ (Brier_perm - Brier_baseline)
        imps.append(r.importances_mean)

    imp_mean = np.mean(imps, axis=0)
    imp_std = np.std(imps, axis=0)

    imp_df = pd.DataFrame({
        "feature": X.columns,
        "delta_brier": imp_mean,   # >0 means feature helps (shuffling hurts)
        "std": imp_std
    }).sort_values("delta_brier", ascending=False)

    return float(np.mean(briers)), imp_df

# ------------------------------------------------- Paths + file names ----------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "submission.csv"

# ------------------------------------------------- CSVs ------------------------------------------------------------- #
# men and women regular season compact results:
MEN_REG = "MRegularSeasonCompactResults.csv"
WOMEN_REG = "WRegularSeasonCompactResults.csv"

# men and women regular season detailed results:
MEN_REG_DET = "MRegularSeasonDetailedResults.csv"
WOMEN_REG_DET = "WRegularSeasonDetailedResults.csv"

# men and women tournament compact results:
MEN_TOUR = "MNCAATourneyCompactResults.csv"
WOMEN_TOUR = "WNCAATourneyCompactResults.csv"

# men and women tournament detailed results:
MEN_TOUR_DET = "MNCAATourneyDetailedResults.csv"
WOMEN_TOUR_DET = "WNCAATourneyDetailedResults.csv"

# men and women seeds:
MEN_SEEDS = "MNCAATourneySeeds.csv"
WOMEN_SEEDS = "WNCAATourneySeeds.csv"

# men and women conferences:
MEN_CONF = "MTeamConferences.csv"
WOMEN_CONF = "WTeamConferences.csv"

# men massey ordinals ranking:
MASSEY_ORDINALS = "MMasseyOrdinals.csv"

# submission csv:
SAMPLE_SUB = "SampleSubmissionStage2.csv"

# ------------------------------------------------- Constants -------------------------------------------------------- #
REG_FINAL_DAY = 132
LAST_TWO_WEEKS_DAY = 110


# ------------------------------------------------- General Helpers -------------------------------------------------- #
# reads the csv in str:
def read_csv(name: str, usecols=None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, usecols=usecols)


# parses the ID column of Season_TeamA_TeamB in the submission file to Season, TeamA, TeamB:
def parse_submission_ids(sub: pd.DataFrame):
    parts = sub["ID"].str.split("_", expand=True)
    season = parts[0].astype(int)
    team_A = parts[1].astype(int)
    team_b = parts[2].astype(int)
    return season, team_A, team_b


# ------------------------------ Loading Data ------------------------------------------------------------------------ #
# adjust game statistics due to overtime
def adjust_overtime(df, is_detailed=False):
    com_columns = ["WScore", "LScore"]
    det_columns = ["LScore", "WScore",
                   "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
                   "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]
    for col in (det_columns if is_detailed else com_columns):
        df[col] = df[col] / ((40 + 5 * df["NumOT"]) / 40)
    return df.drop(columns=["NumOT"])  # after that we have no need for NumOT


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
    loser = loser.drop(columns=[name2 + "Loc"])
    # concatenate and return the result:
    return pd.concat([winner, loser], ignore_index=True)


def get_tournament(is_detailed: bool):
    men_tournament = read_csv(MEN_TOUR_DET if is_detailed else MEN_TOUR)
    women_tournament = read_csv(WOMEN_TOUR_DET if is_detailed else WOMEN_TOUR)
    tournament = pd.concat([men_tournament, women_tournament], ignore_index=True)
    tournament = adjust_overtime(tournament, is_detailed)  # adjust stats to overtime
    tournament = eliminate_winner_loser(tournament, name1="A_", name2="B_")  # swap winner and loser
    tournament["TourDiff"] = tournament["A_Score"] - tournament["B_Score"]
    return tournament


def get_regular(is_detailed: bool):
    men_regular = read_csv(MEN_REG_DET if is_detailed else MEN_REG)
    women_regular = read_csv(WOMEN_REG_DET if is_detailed else WOMEN_REG)
    regular = pd.concat([men_regular, women_regular], ignore_index=True)
    regular = adjust_overtime(regular, is_detailed)  # adjust stats to overtime
    regular = eliminate_winner_loser(regular, name1="", name2="Opp")  # swap winner and loser
    regular["RegDiff"] = regular["Score"] - regular["OppScore"]
    return regular


# ---------------------------- Features ------------------------------------------------------------------------------ #
def get_seeds_feature() -> pd.DataFrame:
    men_seeds = read_csv(MEN_SEEDS)  # 1985 - 2025
    women_seeds = read_csv(WOMEN_SEEDS)  # 1998 - 2025
    seeds = pd.concat([men_seeds, women_seeds], ignore_index=True)
    seeds["Seed"] = np.log(seeds["Seed"].apply(lambda x: int(x[1:3])))
    return seeds


def get_massey_feature() -> pd.DataFrame:
    massey = read_csv(MASSEY_ORDINALS)
    massey = massey[massey["RankingDayNum"] == REG_FINAL_DAY + 1].copy()
    massey = massey[massey["SystemName"].isin(["POM"])]
    consensus = massey.groupby(["Season", "TeamID"], as_index=False).agg(MedianRank=("OrdinalRank", "median"))
    # Custom Log Curve
    consensus["Rank"] = 100 - 4 * np.log(consensus["MedianRank"] + 1) - consensus["MedianRank"] / 22
    return consensus[["Season", "TeamID", "Rank"]]


# returns (Season, TeamID, EloEnd)
def get_elo_feature(
        K=20,
        base=1500,
        new_team_base=1300,
        reversion=0.30,
        hca=75,
        early_season_games=20,
        early_k_boost=0.5
) -> pd.DataFrame:
    def get_conference():
        return pd.concat([read_csv(MEN_CONF), read_csv(WOMEN_CONF)], ignore_index=True)

    def expected_win_prop(rating_A, rating_B) -> float:
        """The expected score / win probability of Team A (rating_A) against Team B (rating_B) """
        return 1.0 / (1 + math.pow(10, (rating_B - rating_A) / 400.0))

    def k_multiplier(games_played):
        # 1.5*K at game 0 -> 1*K at game 20
        g = min(games_played, early_season_games)
        return (1.0 + early_k_boost) - early_k_boost * (g / early_season_games)

    games = pd.concat([read_csv(MEN_REG), read_csv(WOMEN_REG)], ignore_index=True).drop(columns=["NumOT"])
    games = games.sort_values(["Season", "DayNum"], kind="mergesort")
    team_confs = get_conference()
    # Conference map: (Season, TeamID) -> ConfAbbrev
    conf_map = {(int(s), int(t)): c for s, t, c in team_confs[["Season", "TeamID", "ConfAbbrev"]].itertuples(index=False)}
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


def get_team_season_averages(regular):
    averages = regular.groupby(["Season", "TeamID"], as_index=False).mean(numeric_only=True)
    features_list = ["Season", "TeamID", "Score", "OppScore", "W", "RegDiff"]
    return averages[features_list]


def get_team_season_advanced_stats(regular):
    def get_possession(FGA, OR, TO, FTA):
        return (FGA + TO + 0.44 * FTA) - OR

    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return (num / den).fillna(0.0)
    # add needed fields:
    regular["Poss"] = get_possession(regular["FGA"], regular["OR"], regular["TO"], regular["FTA"])
    regular["OppPoss"] = get_possession(regular["OppFGA"], regular["OppOR"], regular["OppTO"], regular["OppFTA"])
    regular["G"] = 1
    # calculate the sums:
    sums = regular.groupby(["Season", "TeamID"], as_index=False).sum(numeric_only=True)
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
    return sums[["Season", "TeamID",
                 "OffEff", "DefEff", "NetEff", "Pace", "eFG", "TOV", "ORB", "FTR", "3PAr", "3P%", "FT%", "TS",
                 "AST_Rate", "STL_Rate", "BLK_Rate", "PF_Rate", "Def_eFG", "Def_3PAr", "Def_3P%", "Def_FTR"]]


def get_team_season_features(is_detailed) -> pd.DataFrame:
    regular = get_regular(is_detailed).drop(columns=["DayNum", "OppTeamID"])
    # build team averages features:
    averages = get_team_season_averages(regular)
    # build team sums features:
    if is_detailed:
        analytics = get_team_season_advanced_stats(regular)
        return averages.merge(analytics, on=["Season", "TeamID"], how="inner")
    else:
        return averages


def get_situational_features() -> pd.DataFrame:
    """Calculates Road Win % and Late Season Win % (Momentum) directly from raw data."""
    df = pd.concat([read_csv(MEN_REG), read_csv(WOMEN_REG)], ignore_index=True)
    # Winner Perspective
    winner = df[["Season", "DayNum", "WTeamID", "WLoc"]].copy()
    winner.columns = ["Season", "DayNum", "TeamID", "Loc"]
    winner["W"] = 1
    # Winner played Away or Neutral
    winner["AwayG"] = winner["Loc"].isin(['A', 'N']).astype(int)
    winner["AwayW"] = winner["Loc"].isin(['A', 'N']).astype(int)
    # Loser Perspective
    loser = df[["Season", "DayNum", "LTeamID", "WLoc"]].copy()
    loser.columns = ["Season", "DayNum", "TeamID", "OppLoc"]
    loser["W"] = 0
    # If the winner played at Home (H) or Neutral (N), the loser played Away/Neutral
    loser["AwayG"] = loser["OppLoc"].isin(['H', 'N']).astype(int)
    loser["AwayW"] = 0

    games = pd.concat([
        winner[["Season", "DayNum", "TeamID", "W", "AwayG", "AwayW"]],
        loser[["Season", "DayNum", "TeamID", "W", "AwayG", "AwayW"]]
    ], ignore_index=True)
    # 1. Road/Neutral Win Percentage
    road_stats = games.groupby(["Season", "TeamID"], as_index=False).agg(
        AwayG=("AwayG", "sum"),
        AwayW=("AwayW", "sum")
    )
    road_stats["Road_WinPct"] = (road_stats["AwayW"] / road_stats["AwayG"].replace(0, np.nan)).fillna(0.0)
    # 2. Late Season Win Percentage (Momentum after Day 110)
    late_games = games[games["DayNum"] >= LAST_TWO_WEEKS_DAY]
    late_stats = late_games.groupby(["Season", "TeamID"], as_index=False).agg(
        Late_G=("W", "count"),
        Late_W=("W", "sum")
    )
    late_stats["Late_WinPct"] = (late_stats["Late_W"] / late_stats["Late_G"].replace(0, np.nan)).fillna(0.0)

    # Combine into one clean dataframe
    feats = road_stats[["Season", "TeamID", "Road_WinPct"]].merge(
        late_stats[["Season", "TeamID", "Late_WinPct"]],
        on=["Season", "TeamID"],
        how="left"
    ).fillna(0.0)

    return feats


def get_features(is_detailed):
    # team-season features:
    reg_season = get_team_season_features(is_detailed)
    # seeds features:
    reg_season = reg_season.merge(get_seeds_feature(), on=["Season", "TeamID"], how="left")
    reg_season["Seed"] = reg_season["Seed"].fillna(np.log(17))
    # elo features:
    reg_season = reg_season.merge(get_elo_feature(), on=["Season", "TeamID"], how="left")
    reg_season = reg_season.merge(get_situational_features(), on=["Season", "TeamID"], how="left")
    # massey features (only for detailed):
    if is_detailed:
        reg_season = reg_season.merge(get_massey_feature(), on=["Season", "TeamID"], how="left")
        reg_season["Rank"] = reg_season["Rank"].fillna(100 - 4 * np.log(175 + 1) - 175 / 22)
        reg_season["Rank"] = np.log(reg_season["Rank"])
    return reg_season


# ---------------------------- Building learning dataset (X,y) ------------------------------------------------------- #
def plot(skeleton):
    tmpmean = skeleton.pivot_table(columns="IsMen", index="EloEnd", values="TourDiff",
                                   aggfunc="mean").sort_index().ffill()
    tmpstd = skeleton.pivot_table(columns="IsMen", index="EloEnd", values="TourDiff",
                                  aggfunc="std").sort_index().ffill().fillna(0)

    fig, axis = plt.subplots(ncols=2, figsize=(12, 4))  # <- no sharey

    (line_1,) = axis[0].plot(tmpmean.index, tmpmean[0], "b-")
    fill_1 = axis[0].fill_between(tmpmean.index, tmpmean[0] - tmpstd[0], tmpmean[0] + tmpstd[0], color="b", alpha=0.1)
    axis[0].set_title("Women")
    axis[0].set_xlabel("SeedDiff")
    axis[0].set_ylabel("TourDiff")

    (line_2,) = axis[1].plot(tmpmean.index, tmpmean[1], "r--")
    fill_2 = axis[1].fill_between(tmpmean.index, tmpmean[1] - tmpstd[1], tmpmean[1] + tmpstd[1], color="r", alpha=0.1)
    axis[1].set_title("Men")
    axis[1].set_xlabel("SeedDiff")

    plt.margins(x=0)
    plt.legend([(line_2, fill_2), (line_1, fill_1)], ["Men", "Women"])
    plt.show()
    exit(1)


def filter_features(features, is_detailed, is_training, to_subtract):
    # Don't touch these attributes:
    excluded = ["Season", "A_TeamID", "B_TeamID"]
    excluded += ["TourDiff"] if is_training else ["ID", "Pred"]
    # evaluate A - B:
    diff = features
    if to_subtract:
        base = [col[2:] for col in features.columns if col.startswith("A_") and col not in excluded]
        diff = pd.DataFrame({k: features[f"A_{k}"] - features[f"B_{k}"] for k in base})
        # add dropped attributes and some new ones:
        diff[excluded] = features[excluded]
    diff["IsMen"] = (diff["A_TeamID"] < 3000).astype(int)
    return diff


def assemble_all_features(skeleton, is_detailed, is_training, to_subtract):
    features = get_features(is_detailed)
    features_A = features.rename(columns={c: f"A_{c}" for c in features.columns if c not in ["Season"]})
    features_B = features.rename(columns={c: f"B_{c}" for c in features.columns if c not in ["Season"]})
    skeleton = skeleton.merge(features_A, on=["Season", "A_TeamID"], how="left")
    skeleton = skeleton.merge(features_B, on=["Season", "B_TeamID"], how="left")
    return filter_features(skeleton, is_detailed, is_training, to_subtract)


def get_training_set(is_detailed, to_subtract):
    tournament = get_tournament(is_detailed)
    tournament = tournament[["Season", "A_TeamID", "B_TeamID", "TourDiff"]]
    training = assemble_all_features(tournament, is_detailed, True, to_subtract)
    training = training.drop(columns=["A_TeamID", "B_TeamID"])  # we have no need for those columns when training
    # leave just the training:
    X = training.drop(columns="TourDiff")
    y = training[["Season", "TourDiff"]]
    return X, y


# ------------------------------------------------- Models ----------------------------------------------------------- #
def train(X, y, model_name):
    models = {}
    off_error = []
    oof_preds = []
    oof_targets = []
    oof_ss = []

    # leave-one-season out models
    for excluded_season in set(X.Season):
        # training (one left out season):
        X_train = X[X.Season != excluded_season].drop(columns=["Season"]).values
        y_train = y[y.Season != excluded_season].drop(columns=["Season"]).TourDiff.values
        # validation:
        X_validation = X[X.Season == excluded_season].drop(columns=["Season"]).values
        y_validation = y[y.Season == excluded_season].drop(columns=["Season"]).TourDiff.values
        # the season:
        s_validation = X[X.Season == excluded_season].Season.values

        if model_name == "xgb":
            # train the model on all included seasons:
            d_train = xgb.DMatrix(X_train, label=y_train)
            model = xgb.train(params=param, dtrain=d_train, num_boost_round=num_rounds)
            models[excluded_season] = model
            # test the model of the excluded season:
            d_validation = xgb.DMatrix(X_validation, label=y_validation)
            predictions_error = model.predict(d_validation)
            error = mean_absolute_error(y_validation, predictions_error)

        else:  # model == "lr"
            # train the model on all included seasons:
            y_train = (y_train > 0).astype(int)
            model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
            model.fit(X_train, y_train)
            models[excluded_season] = model
            # test the model on the excluded season:
            y_validation = (y_validation > 0).astype(int)
            predictions_error = model.predict_proba(X_validation)[:, 1]
            error = brier_score_loss(y_validation, predictions_error)

        print(f"Excluded season: {excluded_season} Brier: {error:.5f}")
        off_error.append(error)
        oof_preds += list(predictions_error)
        oof_targets += list(y_validation)
        oof_ss += list(s_validation)

    print(f"Average mean absolute error: {np.mean(off_error)}")
    labels_1d = (np.array(oof_targets) > 0).astype(int)
    if model_name == "xgb":
        calibrator = LogisticRegression()
        oof_preds_2d = np.array(oof_preds).reshape(-1, 1)
        calibrator.fit(oof_preds_2d, labels_1d)
        xgb_probs = calibrator.predict_proba(oof_preds_2d)[:, 1]
        print(f"Calibrated XGBoost Brier Score: {brier_score_loss(labels_1d, xgb_probs):.5f}")
        return models, calibrator, xgb_probs, labels_1d  # Added returns
    else:  # model == "lr"
        oof_probs = np.array(oof_preds)
        print(f"LR OOF Brier: {brier_score_loss(labels_1d, oof_probs):.5f}")
        return models, None, oof_probs, labels_1d  # Added returns


def predict(is_detailed, models, calibrator, X, model_name, to_subtract, features_to_keep=None):
    # prepare the submission csv:
    sub = read_csv(SAMPLE_SUB, usecols=["ID", "Pred"])
    season, team_A, team_B = parse_submission_ids(sub)
    sub["Season"] = season
    sub["A_TeamID"] = team_A
    sub["B_TeamID"] = team_B
    sub = assemble_all_features(sub, is_detailed, False, to_subtract)

    X_test = sub.drop(columns=["ID", "Pred", "Season", "A_TeamID", "B_TeamID"])

    # --- UPDATED FILTERING LOGIC ---
    if features_to_keep is not None:
        keep_cols = []
        for c in X_test.columns:
            # Strip "A_" or "B_" prefix to check the base name
            base_name = c[2:] if (c.startswith("A_") or c.startswith("B_")) else c
            if base_name in features_to_keep or c == "IsMen":
                keep_cols.append(c)
        X_test = X_test[keep_cols]

    # run models on given dataset
    preds = []
    for excluded_season in set(X.Season):
        model = models[excluded_season]

        if model_name == "xgb":
            dtest = xgb.DMatrix(X_test)
            margin_predictions = model.predict(dtest)
            probs = calibrator.predict_proba(margin_predictions.reshape(-1, 1))[:, 1]
        else:
            probs = model.predict_proba(X_test.values)[:, 1]
        preds.append(probs)

    sub['Pred'] = np.array(preds).mean(axis=0)
    return sub[["ID", "Pred"]]


def get_prediction_from_model(is_detailed, to_subtract, model_name, features_to_keep=None):
    X, y = get_training_set(is_detailed, to_subtract)

    # --- UPDATED FILTERING LOGIC ---
    if features_to_keep is not None:
        keep_cols = []
        for c in X.columns:
            # Strip "A_" or "B_" prefix to check the base name
            base_name = c[2:] if (c.startswith("A_") or c.startswith("B_")) else c
            if base_name in features_to_keep or c in ["Season", "IsMen"]:
                keep_cols.append(c)
        X = X[keep_cols]

    models, calibrator, oof_probs, oof_targets = train(X, y, model_name)
    out = predict(is_detailed, models, calibrator, X, model_name, to_subtract, features_to_keep)
    return out, oof_probs, oof_targets


# ------------------------------------------------- Main ------------------------------------------------------------- #
def main():
    # --------------------------------- Feature Importance ----------------------------------------------------------- #
    print("--- Evaluating Feature Importance (Detailed Data) ---")
    X_feat, y_feat = get_training_set(is_detailed=True, to_subtract=True)
    train_df = pd.concat([X_feat, y_feat["TourDiff"]], axis=1)

    baseline_brier, imp_df = rank_features_by_brier_perm_importance(train_df)

    print(f"\nBaseline XGB Brier for feature evaluation: {baseline_brier:.5f}")
    print("Top 15 Most Important Features:")
    print(imp_df.head(15).to_string(index=False))
    print("-" * 70)

    # Dynamically select only features that improve the Brier score
    good_features = imp_df[imp_df["delta_brier"] > 0.0001]["feature"].tolist()
    print(f"\nSelecting top {len(good_features)} features for Logistic Regression...")
    print("XGBoost will train on ALL available features.")

    # --------------------------------- Prediction & Submission ------------------------------------------------------ #
    print("\n--- Training Detailed LR (Filtered Features) ---")
    # LR gets the strict, clean list of features
    out_det, oof_det, y_targets = get_prediction_from_model(
        is_detailed=True, to_subtract=True, model_name="lr", features_to_keep=good_features
    )

    print("\n--- Training Detailed XGBoost (All Features) ---")
    # XGBoost gets the entire dataset to hunt for non-linear interactions
    out_xg, oof_xg, _ = get_prediction_from_model(
        is_detailed=True, to_subtract=False, model_name="xgb"
    )

    # --------------------------------- Ensemble Optimization -------------------------------------------------------- #
    print("\n--- Optimizing Ensemble Weights ---")
    best_brier = 1.0
    best_w_det = 0.5
    best_w_xg = 0.5

    for w_det in np.arange(0.0, 1.01, 0.01):
        w_xg = 1.0 - w_det

        blend_oof = (w_det * oof_det) + (w_xg * oof_xg)
        brier = brier_score_loss(y_targets, blend_oof)

        if brier < best_brier:
            best_brier = brier
            best_w_det = w_det
            best_w_xg = w_xg

    print(f">> Optimal Blend: {best_w_det:.2f} Detailed LR | {best_w_xg:.2f} Detailed XGB <<")
    print(f">> Final Ensemble Brier Score: {best_brier:.5f} <<")

    # --------------------------------- Apply & Save ----------------------------------------------------------------- #
    out = out_det.copy()
    out["Pred"] = (best_w_det * out_det["Pred"]) + (best_w_xg * out_xg["Pred"])
    out["Pred"] = np.clip(out["Pred"], 1e-5, 1 - 1e-5)

    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
