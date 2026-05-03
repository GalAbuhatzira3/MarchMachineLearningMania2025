# ------------------------------------------------- Imports ---------------------------------------------------------- #
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import xgboost as xgb
import re
import statsmodels.api as sm

# ------------------------------------------------- Settings --------------------------------------------------------- #
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
np.set_printoptions(suppress=True, precision=5)

# ------------------------------------------------- Paths + file names ----------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJECT_ROOT / "data"
OUT_PATH = PROJECT_ROOT / "submission.csv"

# ------------------------------------------------- CSVs ------------------------------------------------------------- #
# men and women regular season compact results:
MEN_REG_COM = "MRegularSeasonCompactResults.csv"
WOMEN_REG_COM = "WRegularSeasonCompactResults.csv"

# men and women regular season detailed results:
MEN_REG_DET = "MRegularSeasonDetailedResults.csv"
WOMEN_REG_DET = "WRegularSeasonDetailedResults.csv"

# men and women tournament compact results:
MEN_TOUR_COM = "MNCAATourneyCompactResults.csv"
WOMEN_TOUR_COM = "WNCAATourneyCompactResults.csv"

# men and women tournament detailed results:
MEN_TOUR_DET = "MNCAATourneyDetailedResults.csv"
WOMEN_TOUR_DET = "WNCAATourneyDetailedResults.csv"

# men and women seeds:
MEN_SEEDS = "MNCAATourneySeeds.csv"
WOMEN_SEEDS = "WNCAATourneySeeds.csv"

# men and women conferences:
MEN_CONF = "MTeamConferences.csv"
WOMEN_CONF = "WTeamConferences.csv"

# submission csv:
SAMPLE_SUB = "SampleSubmissionStage2.csv"

# ------------------------------------------------- Constants -------------------------------------------------------- #
# XGBoost parameters:
param_margin = {}
param_margin["objective"] = "reg:squarederror"
param_margin["booster"] = "gbtree"
param_margin["eta"] = 0.0093
param_margin["subsample"] = 0.6
param_margin["colsample_bynode"] = 0.8
param_margin["num_parallel_tree"] = 2
param_margin["min_child_weight"] = 4
param_margin["max_depth"] = 4
param_margin["tree_method"] = "hist"
param_margin['grow_policy'] = 'lossguide'
param_margin["max_bin"] = 38

# Binary XGBoost parameters:
param_bin = param_margin.copy()
param_bin["objective"] = "binary:logistic"
param_bin["eval_metric"] = "logloss"
num_rounds = 700

# the DayNum that indicates the end of the regular season and the start of the tournament:
REG_FINAL_DAY = 132

# the dayNum that indicates the starting day of the "last two weeks of the regular season", for momentum check:
LAST_TWO_WEEKS_DAY = 110

# the GLM csv will be cached here for performance:
QUALITY_CACHE_FILE = "glm_quality.csv"

# a delta-brier threshold for a feature to be included in LR training:
MODEL_THRESHOLDS = {
    'lr': 0.0001,
    'xgb': 0.003,
    'xgb_bin': 0.0003
}

# men team id are between 0 and 2999 and women's are 3000 and beyond:
MEN_TEAM_ID_THRESHOLD = 3000

# the lowest seed possible (1 = strongest, 16 = weakest)
LOWEST_SEED = 16


# ------------------------------------------------- General Helpers -------------------------------------------------- #
# reads the csv passed in "name"
def read_csv(name: str, usecols=None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, usecols=usecols)


# adjust game statistics due to overtime
def adjust_overtime(df, is_detailed=False):
    com_columns = ["WScore", "LScore"]
    det_columns = ["LScore", "WScore",
                   "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
                   "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF"]
    for col in (det_columns if is_detailed else com_columns):
        df[col] = df[col] / ((40 + 5 * df["NumOT"]) / 40)
    return df.drop(columns=["NumOT"])  # after that we have no need for NumOT


# parses the ID column of Season_TeamA_TeamB in the submission file to Season, TeamA, TeamB
def parse_submission_ids(sub: pd.DataFrame):
    parts = sub["ID"].str.split("_", expand=True)
    season = parts[0].astype(int)
    team_A = parts[1].astype(int)
    team_b = parts[2].astype(int)
    return season, team_A, team_b


# substitute the winner/loser in columns with team/opponent columns
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


# adds a gender column to the dataframe, based on a certain column "col_name"
def add_gender(df, col_name="TeamID"):
    df["IsMen"] = (df[col_name] < MEN_TEAM_ID_THRESHOLD).astype(int)
    return df


# performs permutation importance
def get_oof_permutation_importance(X, y, model_type, n_repeats=30):
    all_importances = []
    seasons = sorted(X["Season"].unique())

    # Identify stems for grouping (A_EloEnd and B_EloEnd -> EloEnd)
    cols = [c for c in X.columns if c != "Season"]
    stems = []
    for c in cols:
        stems.append(c[2:] if (c.startswith("A_") or c.startswith("B_")) else c)
    unique_stems = list(dict.fromkeys(stems))

    print(f"\n--- Calculating Global OOF Importance for: {model_type.upper()} ---")

    for excluded_season in seasons:
        val_mask = (X["Season"] == excluded_season)
        X_tr, X_val = X.loc[~val_mask].drop(columns=["Season"]), X.loc[val_mask].drop(columns=["Season"])
        y_tr_raw, y_val_raw = y.loc[~val_mask, "TourDiff"], y.loc[val_mask, "TourDiff"]

        # Use the same model setup as your train() function
        if model_type == "lr":
            model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
            model.fit(X_tr, (y_tr_raw > 0).astype(int))
            baseline_score = brier_score_loss((y_val_raw > 0).astype(int), model.predict_proba(X_val)[:, 1])
        elif model_type == "xgb":
            model = xgb.XGBRegressor(**param_margin, n_estimators=num_rounds)
            model.fit(X_tr, y_tr_raw)
            baseline_score = mean_absolute_error(y_val_raw, model.predict(X_val))
        else:  # xgb_bin
            model = xgb.XGBClassifier(**param_bin, n_estimators=num_rounds)
            model.fit(X_tr, (y_tr_raw > 0).astype(int))
            baseline_score = brier_score_loss((y_val_raw > 0).astype(int), model.predict_proba(X_val)[:, 1])

        season_importance = {}
        for stem in unique_stems:
            related_cols = [c for c in X_val.columns if c == stem or c == f"A_{stem}" or c == f"B_{stem}"]
            scores = []
            for _ in range(n_repeats):
                X_perm = X_val.copy()
                idx = np.random.permutation(X_perm.index)
                for col in related_cols:
                    X_perm[col] = X_val.loc[idx, col].values

                if model_type == "xgb":
                    score = mean_absolute_error(y_val_raw, model.predict(X_perm))
                else:
                    score = brier_score_loss((y_val_raw > 0).astype(int), model.predict_proba(X_perm)[:, 1])
                scores.append(score - baseline_score)
            season_importance[stem] = np.mean(scores)

        all_importances.append(season_importance)
        print(f"   Season {excluded_season} complete.")

    importance_df = pd.DataFrame(all_importances).mean().sort_values(ascending=False).reset_index()
    importance_df.columns = ["feature", "avg_delta"]

    std_df = pd.DataFrame(all_importances).std().reset_index()
    std_df.columns = ["feature", "std_delta"]
    return importance_df.merge(std_df, on="feature")


# plotting permutation importance results
def plot_importance(imp_df, model_type):
    # Set the title based on model type
    titles = {
        "lr": "Logistic Regression",
        "xgb": "XGBoost",
        "xgb_bin": "XGBoost Binary"
    }
    title = titles.get(model_type, "Model")

    # Get the specific threshold for this model
    threshold = MODEL_THRESHOLDS.get(model_type, 0.0005)

    # Filter for positive importance only
    plot_df = imp_df.sort_values("avg_delta", ascending=True).tail(100)

    if plot_df.empty:
        print(f"No positive importance found for {title}.")
        return

    # Create color list: Red if below threshold, Skyblue if above
    colors = ['red' if val < threshold else 'skyblue' for val in plot_df["avg_delta"]]

    plt.figure(figsize=(10, 8))

    # Plot the bars with the conditional colors
    plt.barh(
        plot_df["feature"],
        plot_df["avg_delta"],
        xerr=plot_df["std_delta"],
        color=colors,
        edgecolor='black'
    )

    # Add the threshold line (vertical, since X represents the delta value)
    plt.axvline(x=threshold, color='gray', linestyle='--', linewidth=1, label=f'Threshold: {threshold}')

    plt.title(f"Permutation Importance: {title}")
    plt.xlabel("Increase in Error (Higher = More Important)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------ Loading Data ------------------------------------------------------------------------ #
# returns the tournament dataframe
def get_tournament(is_detailed: bool):
    men_tournament = read_csv(MEN_TOUR_DET if is_detailed else MEN_TOUR_COM)
    women_tournament = read_csv(WOMEN_TOUR_DET if is_detailed else WOMEN_TOUR_COM)
    tournament = pd.concat([men_tournament, women_tournament], ignore_index=True)
    tournament = adjust_overtime(tournament, is_detailed)  # adjust stats to overtime
    tournament = eliminate_winner_loser(tournament, name1="A_", name2="B_")
    tournament["TourDiff"] = tournament["A_Score"] - tournament["B_Score"]
    return tournament


# returns the regular season dataframe
def get_regular(is_detailed: bool):
    men_regular = read_csv(MEN_REG_DET if is_detailed else MEN_REG_COM)
    women_regular = read_csv(WOMEN_REG_DET if is_detailed else WOMEN_REG_COM)
    regular = pd.concat([men_regular, women_regular], ignore_index=True)
    regular = adjust_overtime(regular, is_detailed)  # adjust stats to overtime
    regular = eliminate_winner_loser(regular, name1="", name2="Opp")
    regular["RegDiff"] = regular["Score"] - regular["OppScore"]
    return regular


# ---------------------------- Features ------------------------------------------------------------------------------ #
def get_seeds_feature() -> pd.DataFrame:
    men_seeds = read_csv(MEN_SEEDS)  # 1985 - 2025
    women_seeds = read_csv(WOMEN_SEEDS)  # 1998 - 2025
    seeds = pd.concat([men_seeds, women_seeds], ignore_index=True)
    seeds["Seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))  # extracting the actual seed
    return seeds


def get_elo_feature(
        k=20,
        base=1500,
        new_team_base=1300,
        reversion=0.30,
        hca=75,
        early_season_games=20,
        early_k_boost=0.5,
        margin_bound=20
) -> pd.DataFrame:

    def expected_win_prop(rating_A, rating_B) -> float:
        """The expected score / win probability of Team A (rating_A) against Team B (rating_B) """
        return 1.0 / (1 + math.pow(10, (rating_B - rating_A) / 400.0))

    def k_multiplier(games_played):
        """1.5*K at game 0 -> 1*K at game 20"""
        g = min(games_played, early_season_games)
        return (1.0 + early_k_boost) - early_k_boost * (g / early_season_games)

    games = pd.concat([read_csv(MEN_REG_COM), read_csv(WOMEN_REG_COM)], ignore_index=True).drop(columns=["NumOT"])
    games = games.sort_values(["Season", "DayNum"], kind="mergesort")
    team_confs = pd.concat([read_csv(MEN_CONF), read_csv(WOMEN_CONF)], ignore_index=True)
    # Conference map: (Season, TeamID) -> ConfAbbrev
    conf_map = {(int(s), int(t)): c for s, t, c in team_confs.itertuples(index=False)}
    rows = []
    seasons = games["Season"].unique()
    prev_elo = {}  # (TeamID, rating) dict of last season
    for season in seasons:  # for every season 1985 until 2024:
        season_games = games[games["Season"] == season]  # take just the games of this season
        active_teams = set(season_games["WTeamID"]).union(set(season_games["LTeamID"]))  # teams that played this season
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
            margin = min(margin_bound, WScore - LScore)
            mov_multiplier = np.log(margin) + 1.0

            # APPLYING: Early-season K - use average of the two teams' multipliers
            k_eff = 0.5 * k * (k_multiplier(game_played[WTeamID]) + k_multiplier(game_played[LTeamID]))

            # Reevaluating the ratings of both winner and loser based on this single game:
            delta = k_eff * (1.0 - expected_winner) * mov_multiplier  # winner S=1
            ratings[WTeamID] += delta
            ratings[LTeamID] -= delta

            game_played[WTeamID] += 1
            game_played[LTeamID] += 1
        # Save end-of-season for reversion
        for teamID in active_teams:
            rows.append((int(season), teamID, float(ratings[teamID])))
        prev_elo = {int(t): float(ratings[int(t)]) for t in active_teams}

    elo = pd.DataFrame(rows, columns=["Season", "TeamID", "EloEnd"])
    elo = elo.sort_values(["Season", "TeamID"]).drop_duplicates(["Season", "TeamID"], keep="last")

    return elo


def get_team_season_averages(regular):
    averages = regular.groupby(["Season", "TeamID"], as_index=False).mean(numeric_only=True)
    # picking only those due to the fear of correlation with the advanced stats:
    features_list = ["Season", "TeamID", "Score", "OppScore", "W", "RegDiff"]
    return averages[features_list]


def get_team_season_advanced_stats(regular):
    def get_possession(FGA, OR, TO, FTA):
        return (FGA + TO + 0.44 * FTA) - OR

    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return (num / den).fillna(0.0)

    # ass possessions fields:
    regular["Poss"] = get_possession(regular["FGA"], regular["OR"], regular["TO"], regular["FTA"])
    regular["OppPoss"] = get_possession(regular["OppFGA"], regular["OppOR"], regular["OppTO"], regular["OppFTA"])
    # Efficiency fields:
    regular["OffEff"] = safe_div(regular["Score"], regular["Poss"])
    regular["DefEff"] = safe_div(regular["OppScore"], regular["OppPoss"])
    regular["NetEff"] = regular["OffEff"] - regular["DefEff"]
    # Dean Oliver's Four Factors:
    regular["eFG"] = safe_div(regular["FGM"] + 0.5 * regular["FGM3"], regular["FGA"])
    regular["TOV"] = safe_div(regular["TO"], regular["Poss"])
    regular["ORB"] = safe_div(regular["OR"], regular["OR"] + regular["OppDR"])
    regular["FTR"] = safe_div(regular["FTA"], regular["FGA"])

    regular["3PAr"] = safe_div(regular["FGA3"], regular["FGA"])
    regular["3P%"] = safe_div(regular["FGM3"], regular["FGA3"])
    regular["FT%"] = safe_div(regular["FTM"], regular["FTA"])
    regular["TS"] = safe_div(regular["Score"], 2.0 * (regular["FGA"] + 0.44 * regular["FTA"]))

    regular["AST_Rate"] = safe_div(regular["Ast"], regular["Poss"])
    regular["STL_Rate"] = safe_div(regular["Stl"], regular["OppPoss"])
    regular["BLK_Rate"] = safe_div(regular["Blk"], regular["OppFGA"])
    regular["PF_Rate"] = safe_div(regular["PF"], regular["Poss"])

    regular["Def_eFG"] = safe_div(regular["OppFGM"] + 0.5 * regular["OppFGM3"], regular["OppFGA"])
    regular["Def_3PAr"] = safe_div(regular["OppFGA3"], regular["OppFGA"])
    regular["Def_3P%"] = safe_div(regular["OppFGM3"], regular["OppFGA3"])
    regular["Def_FTR"] = safe_div(regular["OppFTA"], regular["OppFGA"])
    analytics = regular.groupby(["Season", "TeamID"], as_index=False).mean(numeric_only=True)
    analytics = analytics[["Season", "TeamID",
                           "OffEff", "DefEff", "NetEff", "eFG", "TOV", "ORB", "FTR", "Poss", "3PAr", "3P%", "FT%", "TS",
                           "AST_Rate", "STL_Rate", "BLK_Rate", "PF_Rate", "Def_eFG", "Def_3PAr", "Def_3P%", "Def_FTR"]]
    return analytics


def get_team_season_features(is_detailed) -> pd.DataFrame:
    regular = get_regular(is_detailed).drop(columns=["DayNum", "OppTeamID"])
    # build team averages features:
    averages = get_team_season_averages(regular)
    if not is_detailed:
        return averages
    # build team sums features (advanced possessions-based analytics):
    analytics = get_team_season_advanced_stats(regular)
    return averages.merge(analytics, on=["Season", "TeamID"], how="inner")


def get_situational_features() -> pd.DataFrame:
    """Calculates Road Win % and Late Season Win % (Momentum) directly from raw data."""
    df = pd.concat([read_csv(MEN_REG_COM), read_csv(WOMEN_REG_COM)], ignore_index=True)
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
    features = road_stats[["Season", "TeamID", "Road_WinPct"]].merge(
        late_stats[["Season", "TeamID", "Late_WinPct"]],
        on=["Season", "TeamID"],
        how="left"
    ).fillna(0.0)

    return features


def get_quality_feature(cache: bool = True) -> pd.DataFrame:
    processed_dir = PROJECT_ROOT / "cache"
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / QUALITY_CACHE_FILE

    if cache and path.exists():
        q = pd.read_csv(path)
        return q[["Season", "TeamID", "Quality"]]

    # seeds (we only need Season, TeamID)
    seeds = get_seeds_feature()
    seeds = add_gender(seeds)
    regular = get_regular(is_detailed=False)
    regular = add_gender(regular)

    def extract_team_id(param_name: str) -> int | None:
        # patsy names look like: "T1_TeamID[1181]"  (sometimes also "T1_TeamID[T.1181]" depending on encoding)
        m = re.search(r"TeamID\[(?:T\.)?(\d+)\]", param_name)
        return int(m.group(1)) if m else None

    out_rows = []
    seasons = sorted(seeds["Season"].unique())

    for season in seasons:
        for is_men in (0, 1):
            # a set of the teams that participated in the tournament that season:
            tour_teams = set(seeds[(seeds["Season"] == season) & (seeds["IsMen"] == is_men)].TeamID.astype(int))
            if len(tour_teams) == 0:
                continue
            # just the regular season stats for that season:
            reg_season = regular[(regular["Season"] == season) & (regular["IsMen"] == is_men)]
            if reg_season.empty:
                continue
            # teams that didn't make the tournament, but beat a tournament team at least once in the regular season:
            upset_teams = set(reg_season[(reg_season.Score > reg_season.OppScore) &
                                         (~reg_season.TeamID.isin(tour_teams)) &
                                         (reg_season.OppTeamID.isin(tour_teams))].TeamID.astype(int))
            # union:
            teams = tour_teams.union(upset_teams)

            # keep games where at least one side is relevant, collapse the other side to 0000
            reg_season = reg_season[["TeamID", "OppTeamID", "RegDiff"]]
            relevant_games = reg_season[reg_season.TeamID.isin(teams) | reg_season.OppTeamID.isin(teams)].copy()
            if relevant_games.empty:
                continue

            relevant_games.loc[~relevant_games["TeamID"].isin(teams), "TeamID"] = 0
            relevant_games.loc[~relevant_games["OppTeamID"].isin(teams), "OppTeamID"] = 0

            relevant_games["TeamID"] = relevant_games["TeamID"].astype(int).astype(str).str.zfill(4)
            relevant_games["OppTeamID"] = relevant_games["OppTeamID"].astype(int).astype(str).str.zfill(4)

            try:
                glm = sm.GLM.from_formula(
                    "RegDiff ~ -1 + TeamID + OppTeamID",
                    data=relevant_games,
                    family=sm.families.Gaussian(),
                ).fit()
            except Exception as e:
                continue

            for name, val in glm.params.items():
                team_id = extract_team_id(name)
                if team_id is None or team_id == 0:
                    continue
                out_rows.append((int(season), int(team_id), int(is_men), float(val)))

    quality = pd.DataFrame(out_rows, columns=["Season", "TeamID", "IsMen", "Quality"])

    # center within season+gender (stabilizes raw A_/B_ usage; diffs are unchanged anyway)
    quality["Quality"] = quality["Quality"] - quality.groupby(["Season", "IsMen"])["Quality"].transform("mean")

    quality = quality[["Season", "TeamID", "Quality"]].drop_duplicates(["Season", "TeamID"], keep="last")
    quality = quality.sort_values(["Season", "TeamID"]).reset_index(drop=True)
    print(quality)

    if cache:
        quality.to_csv(path, index=False)

    return quality


def get_features(is_detailed):
    # team-season features:
    features = get_team_season_features(is_detailed)
    # seeds feature:
    features = features.merge(get_seeds_feature(), on=["Season", "TeamID"], how="left")
    features["Seed"] = features["Seed"].fillna(LOWEST_SEED + 1)  # avoids nan values
    # Women’s games are significantly more "top-heavy" than Men's. A 1-seed in the Women's tournament is statistically
    # more likely to win than a 1-seed in the Men's.
    features["Seed_Gender_Interact"] = features["Seed"] * (1 - (features["TeamID"] < MEN_TEAM_ID_THRESHOLD).astype(int))
    # elo feature:
    features = features.merge(get_elo_feature(), on=["Season", "TeamID"], how="left")
    # road win pct and late-season win pct features:
    features = features.merge(get_situational_features(), on=["Season", "TeamID"], how="left")
    # quality feature:
    quality_df = get_quality_feature()
    features = features.merge(quality_df, on=["Season", "TeamID"], how="left")
    features["Quality"] = features["Quality"].fillna(0.0)  # avoids nan values
    return features


# ---------------------------- Building learning dataset (X,y) ------------------------------------------------------- #
def filter_features(features, is_training, to_subtract):
    # core attributes:
    excluded = ["Season", "A_TeamID", "B_TeamID"]
    excluded += ["TourDiff"] if is_training else ["ID", "Pred"]
    # evaluate A - B if required:
    diff = features
    if to_subtract:
        base = [col[2:] for col in features.columns if col.startswith("A_") and col not in excluded]
        diff = pd.DataFrame({k: features[f"A_{k}"] - features[f"B_{k}"] for k in base})
        # add dropped attributes and some new ones:
        diff[excluded] = features[excluded]
    diff = add_gender(diff, col_name="A_TeamID")
    return diff


def assemble_all_features(skeleton, is_detailed, is_training, to_subtract):
    features = get_features(is_detailed)
    # renaming:
    features_A = features.rename(columns={c: f"A_{c}" for c in features.columns if c not in ["Season"]})
    features_B = features.rename(columns={c: f"B_{c}" for c in features.columns if c not in ["Season"]})
    # connect the features for Team A:
    skeleton = skeleton.merge(features_A, on=["Season", "A_TeamID"], how="left")
    # connect the features for Team B:
    skeleton = skeleton.merge(features_B, on=["Season", "B_TeamID"], how="left")
    return filter_features(skeleton, is_training, to_subtract)


def get_training_set(is_detailed, to_subtract):
    tournament = get_tournament(is_detailed)
    tournament = tournament[["Season", "A_TeamID", "B_TeamID", "TourDiff"]]  # no need for the rest of the attributes
    training = assemble_all_features(tournament, is_detailed, True, to_subtract)
    training = training.drop(columns=["A_TeamID", "B_TeamID"])  # we have no need for those columns while training
    # leave just the training:
    X = training.drop(columns="TourDiff")
    y = training[["Season", "TourDiff"]]
    return X, y


# ------------------------------------------------- Models ----------------------------------------------------------- #
def train(X, y, model_name):
    """Leave-one-subject-out training.

    model_name:
        - 'lr'      : Logistic Regression on (A-B) features (probabilities)
        - 'xgb'     : XGBoost regressor on TourDiff margins + Platt calibration to probabilities
        - 'xgb_bin' : XGBoost binary:logistic directly predicting probabilities

    Returns:
        models: dict[excluded_season -> fitted model]
        calibrator: fitted calibrator for xgb margins (only for model_name='xgb'), else None
        oof_probs: np.ndarray of probabilities aligned to X.index order
        labels: np.ndarray of 0/1 labels aligned to X.index order
    """
    print("Training and Validation...")
    models = {}
    seasons = sorted(X["Season"].unique())

    # Aligned OOF containers
    oof_pred = pd.Series(index=X.index, dtype=float)
    oof_label = pd.Series(index=X.index, dtype=int)

    fold_scores = []
    i = 1
    for excluded_season in seasons:
        val_mask = (X["Season"] == excluded_season)
        val_idx = X.index[val_mask]
        # training: X and y without the excluded season:
        X_train = X.loc[~val_mask].drop(columns=["Season"]).values
        y_train_raw = y.loc[~val_mask, "TourDiff"].values
        y_train = (y_train_raw > 0).astype(int)  # convert to binary
        # validation: X and y of the excluded season:
        X_val = X.loc[val_mask].drop(columns=["Season"]).values
        y_val_raw = y.loc[val_mask, "TourDiff"].values
        y_val = (y_val_raw > 0).astype(int)  # convert to binary

        if model_name == "xgb":
            # Margin regression (TourDiff) then Platt to win probability
            model = xgb.train(params=param_margin, dtrain=xgb.DMatrix(X_train, label=y_train_raw), num_boost_round=num_rounds)
            models[excluded_season] = model

            margins = model.predict(xgb.DMatrix(X_val, label=y_val_raw))

            oof_pred.loc[val_idx] = margins
            oof_label.loc[val_idx] = (y_val_raw > 0).astype(int)

            mae = mean_absolute_error(y_val_raw, margins)
            fold_scores.append(mae)
            # print(f"Excluded season: {excluded_season}. MAE: {mae:.5f}")

        elif model_name == "xgb_bin":
            model = xgb.train(params=param_bin, dtrain=xgb.DMatrix(X_train, label=y_train), num_boost_round=num_rounds)
            models[excluded_season] = model

            probs = model.predict(xgb.DMatrix(X_val, label=y_val))
            probs = np.clip(probs, 1e-6, 1 - 1e-6)

            oof_pred.loc[val_idx] = probs
            oof_label.loc[val_idx] = y_val

            brier = brier_score_loss(y_val, probs)
            fold_scores.append(brier)
            # print(f"Excluded season: {excluded_season}. Brier: {brier:.5f}")

        else:  # lr
            # training and saving a new model:
            model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
            model.fit(X_train, y_train)
            models[excluded_season] = model
            # making a prediction for the validation X:
            probs = model.predict_proba(X_val)[:, 1]
            # storing those predictions for the excluded season:
            oof_pred.loc[val_idx] = probs
            # storing the real results for the excluded season:
            oof_label.loc[val_idx] = y_val
            # computing the brier score of those predictions and adding it:
            brier = brier_score_loss(y_val, probs)
            fold_scores.append(brier)
            # print(f"Excluded season: {excluded_season}. Brier: {brier:.5f}")
        print(f"\r[{i}/{len(seasons)}] Models have completed training and validation", end="", flush=True)
        i += 1
    # those are the y labels for all the ~4500 games:
    labels = oof_label.astype(int).values

    if model_name == "xgb":
        margins = oof_pred.astype(float).values
        calibrator = LogisticRegression(max_iter=2000)
        calibrator.fit(margins.reshape(-1, 1), labels)
        probs = calibrator.predict_proba(margins.reshape(-1, 1))[:, 1]
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        print(f"\nAverage MAE (per season): {float(np.mean(fold_scores)):.5f}")
        print(f"Total Calibrated Brier (across all games): {brier_score_loss(labels, probs):.5f}")
        print("✅ Done!")
        return models, calibrator, probs, labels
    else:
        probs = np.clip(oof_pred.astype(float).values, 1e-6, 1 - 1e-6)
        print(f"\nAverage Brier (per season): {float(np.mean(fold_scores)):.5f}")
        print(f"Total Brier (across all games): {brier_score_loss(labels, probs):.5f}")
        print("✅ Done!")
        return models, None, probs, labels


def predict(is_detailed, models, calibrator, X, model_name, to_subtract, features_to_keep=None):
    print("Predicting...")
    # prepare the submission csv:
    sub = read_csv(SAMPLE_SUB, usecols=["ID", "Pred"])
    season, team_A, team_B = parse_submission_ids(sub)
    sub["Season"] = season
    sub["A_TeamID"] = team_A
    sub["B_TeamID"] = team_B
    sub = assemble_all_features(sub, is_detailed, False, to_subtract)
    X_test = sub.drop(columns=["ID", "Pred", "Season", "A_TeamID", "B_TeamID"])

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
    i = 1
    seasons = sorted(X.Season.unique())
    for excluded_season in seasons:
        model = models[excluded_season]
        if model_name == "xgb":
            margin_predictions = model.predict(xgb.DMatrix(X_test))
            probs = calibrator.predict_proba(margin_predictions.reshape(-1, 1))[:, 1]
        elif model_name == "xgb_bin":
            probs = model.predict(xgb.DMatrix(X_test))
        else:  # Logistic Regression
            probs = model.predict_proba(X_test.values)[:, 1]
        preds.append(probs)
        print(f"\r[{i}/{len(models)}] Models generated predictions", end="", flush=True)
        i += 1
    sub['Pred'] = np.array(preds).mean(axis=0)
    print("\nTaking the mean prediction for every match")
    print("✅ Done!")
    return sub[["ID", "Pred"]]


def get_prediction_from_model(is_detailed, to_subtract, model_name, features_to_keep=None):
    X, y = get_training_set(is_detailed, to_subtract)

    if features_to_keep is not None:
        keep_cols = []
        for c in X.columns:
            base_name = c[2:] if (c.startswith("A_") or c.startswith("B_")) else c
            if base_name in features_to_keep or c in ["Season", "IsMen"]:
                keep_cols.append(c)
        X = X[keep_cols]

    models, calibrator, oof_probs, oof_targets = train(X, y, model_name)
    out = predict(is_detailed, models, calibrator, X, model_name, to_subtract, features_to_keep)
    return out, oof_probs, oof_targets


def get_best_features(model_type):
    to_subtract = True if model_type == 'lr' else False
    X, y = get_training_set(is_detailed=True, to_subtract=to_subtract)
    lr_importance = get_oof_permutation_importance(X, y, model_type=model_type)

    plot_importance(lr_importance, model_type)
    threshold = MODEL_THRESHOLDS.get(model_type, 0.0005)
    best_features = lr_importance[lr_importance["avg_delta"] >= threshold]["feature"].tolist()
    print(lr_importance)
    print(best_features)
    return best_features


def main():
    # --------------------------------- Prediction & Submission ------------------------------------------------------ #
    print("Training base model #1: Logistic Regression (filtered features, A-B diffs, detailed):")
    features = ['EloEnd', 'Quality', 'OffEff', 'TS', 'NetEff', 'Def_eFG', 'FTR', 'Score', 'PF_Rate', 'Def_FTR',
                'Seed_Gender_Interact', 'Poss', 'eFG', 'BLK_Rate', '3P%', 'AST_Rate', 'OppScore', 'RegDiff',
                'TOV', 'Def_3P%', '3PAr', 'Road_WinPct', 'W', 'ORB', 'DefEff']
    # features = get_best_features(model_type='lr')
    out_detailed, oof_detailed, y_targets = get_prediction_from_model(
        is_detailed=True, to_subtract=True, model_name="lr", features_to_keep=features
    )

    print("\nTraining base model #2: XGBoost margin regression (calibrated probability):")
    features = ['EloEnd', 'Seed', 'Quality', 'Seed_Gender_Interact', 'NetEff', 'OffEff', 'ORB', 'AST_Rate', 'Score',
                'OppScore', 'RegDiff', 'Def_3P%', 'STL_Rate', 'FT%', '3P%']
    # features = get_best_features(model_type='xgb')
    out_xg, oof_xg, _ = get_prediction_from_model(
        is_detailed=True, to_subtract=False, model_name="xgb", features_to_keep=features
    )

    print("\nTraining base model #3: XGBoost (binary logistic):")
    features = ['Seed', 'EloEnd', 'Quality', 'FTR', 'OffEff', 'Seed_Gender_Interact', 'Score']
    # features = get_best_features(model_type='xgb_bin')
    out_xg_bin, oof_xg_bin, _ = get_prediction_from_model(
        is_detailed=True, to_subtract=False, model_name="xgb_bin", features_to_keep=features
    )

    # out_ = each model's predictions for the 2025 season.
    # oof_ = the oof predictions for each of the training games (2003 season - 2024 season)

    # --------------------------------- Stacking ----------------------------------------------------------- #
    # in this part, due to us having multiple models, we perform "stacking" to determine the weights of the prediction
    # each model should have based on history:
    print("\nStacking (Meta Logistic) ---")

    # For meta training, we want IsMen aligned with the LR training rows
    # (same tournament rows/order as oof_det / y_targets thanks to index-aligned OOF).
    X_meta_base, _ = get_training_set(is_detailed=True, to_subtract=True)
    is_men_train = X_meta_base["IsMen"].astype(int).values

    # building the meta training features:
    meta_feats = np.column_stack([oof_detailed, oof_xg, oof_xg_bin])
    print("Meta model has been created")

    # Fitting two metamodels (Men/Women) on OOF predictions:
    meta_men = LogisticRegression(max_iter=2000)
    meta_women = LogisticRegression(max_iter=2000)

    # building the masks for splitting rows to men and women:
    men_mask = (is_men_train == 1)
    women_mask = (is_men_train == 0)

    # fitting models using OOF base predictions:
    print("Training...")
    meta_men.fit(meta_feats[men_mask], y_targets[men_mask])
    meta_women.fit(meta_feats[women_mask], y_targets[women_mask])
    print("✅ Done!")

    print("Validation...")
    oof_meta = np.empty_like(y_targets, dtype=float)
    # generate probabilities from the metamodels for the same training rows:
    oof_meta[men_mask] = meta_men.predict_proba(meta_feats[men_mask])[:, 1]
    oof_meta[women_mask] = meta_women.predict_proba(meta_feats[women_mask])[:, 1]
    oof_meta = np.clip(oof_meta, 1e-6, 1 - 1e-6)
    print(f"OOF Meta Brier: {brier_score_loss(y_targets, oof_meta):.5f}")
    print("✅ Done!")

    # --------------------------------- Apply Meta & Save ------------------------------------------------------------ #
    print("Predicting...")
    out = out_detailed.copy()

    # Build meta features for submission
    # ID format is SSSS_XXXX_YYYY and XXXX is the LOWER TeamID (A_TeamID in our parsing)
    season_sub, teamA_sub, teamB_sub = parse_submission_ids(out)
    is_men_sub = (teamA_sub < MEN_TEAM_ID_THRESHOLD).astype(int)

    meta_sub = np.column_stack([out_detailed["Pred"].values, out_xg["Pred"].values, out_xg_bin["Pred"].values])

    final_pred = np.empty(len(out), dtype=float)
    men_mask_sub = (is_men_sub == 1)
    wom_mask_sub = (is_men_sub == 0)
    final_pred[men_mask_sub] = meta_men.predict_proba(meta_sub[men_mask_sub])[:, 1]
    final_pred[wom_mask_sub] = meta_women.predict_proba(meta_sub[wom_mask_sub])[:, 1]

    out["Pred"] = np.clip(final_pred, 1e-5, 1 - 1e-5)

    # Kaggle winning strategy:
    # out["Pred"] = np.where(out["Pred"] >= 0.8, 1.0, out["Pred"])
    # out["Pred"] = np.where(out["Pred"] <= 0.2, 0.0, out["Pred"])

    out.to_csv(OUT_PATH, index=False)
    print("✅ Done!")
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
