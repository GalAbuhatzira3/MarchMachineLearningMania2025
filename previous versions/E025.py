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
from sklearn.metrics import log_loss
import xgboost as xgb
import re
import statsmodels.api as sm


def neg_brier_scorer(estimator, X, y):
    p = estimator.predict_proba(X)[:, 1]
    return -brier_score_loss(y, p)  # higher is better


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

# Binary XGB for direct probability training
param_bin = param_margin.copy()
param_bin["objective"] = "binary:logistic"
param_bin["eval_metric"] = "logloss"

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
            scoring=scorer,  # negative Brier
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
        "delta_brier": imp_mean,  # >0 means feature helps (shuffling hurts)
        "std": imp_std
    }).sort_values("delta_brier", ascending=False)

    return float(np.mean(briers)), imp_df


# ------------------------------------------------- Paths + file names ----------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
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

# men massey ordinals ranking:
MASSEY_ORDINALS = "MMasseyOrdinals.csv"

# men and women tournament slots:
MEN_TOUR_SLOT = "MNCAATourneySlots.csv"
WOMEN_TOUR_SLOT = "WNCAATourneySlots.csv"

# submission csv:
SAMPLE_SUB = "SampleSubmissionStage2.csv"

# ------------------------------------------------- Constants -------------------------------------------------------- #
# the DayNum that indicates the end of the regular season and the start of the tournament:
REG_FINAL_DAY = 132

# the dayNum that indicates the starting day of the "last two weeks of the regular season", for momentum check:
LAST_TWO_WEEKS_DAY = 110

# the GLM csv will be cached here for performance:
QUALITY_CACHE_FILE = "glm_quality.csv"

# a delta-brier threshold for a feature to be included in LR training:
FEATURE_IMPORTANCE_THRESHOLD = 0.0005

# DayNum the massey ordinals are published:
MASSEY_DAY = 133

# reliable massey ordinals systems:
MASSEY_SYSTEMS = ['POM', 'SAG', 'COL', 'DOL', 'MOR', 'WLK', 'RTH']

# men team id are between 0 and 2999 and women's are 3000 and beyond:
MEN_TEAM_ID_THRESHOLD = 3000

# the lowest seed possible (1 = strongest, 16 = weakest)
LOWEST_SEED = 16


# ------------------------------------------------- General Helpers -------------------------------------------------- #
# reads the csv in str:
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


# parses the ID column of Season_TeamA_TeamB in the submission file to Season, TeamA, TeamB:
def parse_submission_ids(sub: pd.DataFrame):
    parts = sub["ID"].str.split("_", expand=True)
    season = parts[0].astype(int)
    team_A = parts[1].astype(int)
    team_b = parts[2].astype(int)
    return season, team_A, team_b


# picks the best features to be used in logistic regression:
def get_good_features():
    print("Evaluating Feature Importance (Detailed Data):")
    X_feat, y_feat = get_training_set(is_detailed=True, to_subtract=True)
    train_df = pd.concat([X_feat, y_feat["TourDiff"]], axis=1)

    baseline_brier, imp_df = rank_features_by_brier_perm_importance(train_df)

    print(f"\nBaseline XGB Brier for feature evaluation: {baseline_brier:.5f}")
    print("Top 15 Most Important Features:")
    print(imp_df.head(15).to_string(index=False))
    print("-" * 70)

    # Dynamically select only features that improve the Brier score
    good_features = imp_df[imp_df["delta_brier"] > FEATURE_IMPORTANCE_THRESHOLD]["feature"].tolist()
    print(f"\nSelecting top {len(good_features)} features for Logistic Regression...")
    return good_features


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


def add_gender(df, col_name="TeamID"):
    df["IsMen"] = (df[col_name] < MEN_TEAM_ID_THRESHOLD).astype(int)
    return df


# ------------------------------ Loading Data ------------------------------------------------------------------------ #
def get_tournament(is_detailed: bool):
    men_tournament = read_csv(MEN_TOUR_DET if is_detailed else MEN_TOUR_COM)
    women_tournament = read_csv(WOMEN_TOUR_DET if is_detailed else WOMEN_TOUR_COM)
    tournament = pd.concat([men_tournament, women_tournament], ignore_index=True)
    tournament = adjust_overtime(tournament, is_detailed)  # adjust stats to overtime
    tournament = eliminate_winner_loser(tournament, name1="A_", name2="B_")
    tournament["TourDiff"] = tournament["A_Score"] - tournament["B_Score"]
    return tournament


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


# returns (Season, TeamID, EloEnd)
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
        # 1.5*K at game 0 -> 1*K at game 20
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
                 "OffEff", "DefEff", "NetEff", "eFG", "TOV", "ORB", "FTR", "Pace", "3PAr", "3P%", "FT%", "TS",
                 "AST_Rate", "STL_Rate", "BLK_Rate", "PF_Rate", "Def_eFG", "Def_3PAr", "Def_3P%", "Def_FTR"]]


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
    processed_dir = PROJECT_ROOT / "data" / "processed"
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
    features["Seed"] = features["Seed"].fillna(LOWEST_SEED + 1)
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
    features["Quality"] = features["Quality"].fillna(0.0)
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
    """Leave-one-season-out training with INDEX-ALIGNED OOF predictions.

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
    models = {}
    seasons = sorted(X["Season"].unique())

    # Aligned OOF containers
    oof_pred = pd.Series(index=X.index, dtype=float)
    oof_label = pd.Series(index=X.index, dtype=int)

    fold_scores = []

    for excluded_season in seasons:
        val_mask = (X["Season"] == excluded_season)
        val_idx = X.index[val_mask]

        X_train = X.loc[~val_mask].drop(columns=["Season"]).values
        y_train_raw = y.loc[~val_mask, "TourDiff"].values

        X_val = X.loc[val_mask].drop(columns=["Season"]).values
        y_val_raw = y.loc[val_mask, "TourDiff"].values

        if model_name == "xgb":
            # Margin regression (TourDiff) then Platt to win probability
            model = xgb.train(params=param_margin, dtrain=xgb.DMatrix(X_train, label=y_train_raw), num_boost_round=num_rounds)
            models[excluded_season] = model

            margins = model.predict(xgb.DMatrix(X_val, label=y_val_raw))

            oof_pred.loc[val_idx] = margins
            oof_label.loc[val_idx] = (y_val_raw > 0).astype(int)

            mae = mean_absolute_error(y_val_raw, margins)
            fold_scores.append(mae)
            print(f"Excluded season: {excluded_season} MAE: {mae:.5f}")

        elif model_name == "xgb_bin":
            # Direct probability model
            y_train = (y_train_raw > 0).astype(int)
            y_val = (y_val_raw > 0).astype(int)

            model = xgb.train(params=param_bin, dtrain=xgb.DMatrix(X_train, label=y_train), num_boost_round=num_rounds)
            models[excluded_season] = model

            probs = model.predict(xgb.DMatrix(X_val, label=y_val))
            probs = np.clip(probs, 1e-6, 1 - 1e-6)

            oof_pred.loc[val_idx] = probs
            oof_label.loc[val_idx] = y_val

            brier = brier_score_loss(y_val, probs)
            fold_scores.append(brier)
            print(f"Excluded season: {excluded_season} Brier: {brier:.5f}")

        else:  # lr
            y_train = (y_train_raw > 0).astype(int)
            model = Pipeline([("scaler", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000))])
            model.fit(X_train, y_train)
            models[excluded_season] = model

            y_val = (y_val_raw > 0).astype(int)
            probs = model.predict_proba(X_val)[:, 1]

            oof_pred.loc[val_idx] = probs
            oof_label.loc[val_idx] = y_val

            brier = brier_score_loss(y_val, probs)
            fold_scores.append(brier)
            print(f"Excluded season: {excluded_season} Brier: {brier:.5f}")

    labels = oof_label.astype(int).values

    if model_name == "xgb":
        margins = oof_pred.astype(float).values
        calibrator = LogisticRegression(max_iter=2000)
        calibrator.fit(margins.reshape(-1, 1), labels)
        probs = calibrator.predict_proba(margins.reshape(-1, 1))[:, 1]
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        print(f"Average MAE: {float(np.mean(fold_scores)):.5f}")
        print(f"OOF XGB calibrated Brier: {brier_score_loss(labels, probs):.5f}")

        return models, calibrator, probs, labels

    if model_name == "xgb_bin":
        probs = np.clip(oof_pred.astype(float).values, 1e-6, 1 - 1e-6)
        print(f"Average Brier: {float(np.mean(fold_scores)):.5f}")
        print(f"OOF XGB_bin Brier: {brier_score_loss(labels, probs):.5f}")
        return models, None, probs, labels

    # lr
    probs = np.clip(oof_pred.astype(float).values, 1e-6, 1 - 1e-6)
    print(f"Average Brier: {float(np.mean(fold_scores)):.5f}")
    print(f"LR OOF Brier: {brier_score_loss(labels, probs):.5f}")

    return models, None, probs, labels


def predict(is_detailed, models, calibrator, X, model_name, to_subtract, features_to_keep=None):
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
    for excluded_season in sorted(X.Season.unique()):
        model = models[excluded_season]

        if model_name == "xgb":
            margin_predictions = model.predict(xgb.DMatrix(X_test))
            probs = calibrator.predict_proba(margin_predictions.reshape(-1, 1))[:, 1]
        elif model_name == "xgb_bin":
            probs = model.predict(xgb.DMatrix(X_test))
        else:
            probs = model.predict_proba(X_test.values)[:, 1]
        preds.append(probs)

    sub['Pred'] = np.array(preds).mean(axis=0)
    return sub[["ID", "Pred"]]


def get_prediction_from_model(is_detailed, to_subtract, model_name, features_to_keep=None):
    X, y = get_training_set(is_detailed, to_subtract)

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
    # at this part we're picking the features that will be used for our logistic regression model
    good_features = get_good_features()
    exit(1)

    # --------------------------------- Prediction & Submission ------------------------------------------------------ #
    print("\nTraining base model #1: Logistic Regression (filtered features, A-B diffs, detailed):")
    # LR gets the strict, clean list of features
    out_detailed, oof_detailed, y_targets = get_prediction_from_model(
        is_detailed=True, to_subtract=True, model_name="lr", features_to_keep=good_features
    )

    print("\nTraining base model #2: XGBoost margin regression (calibrated probability):")
    # XGBoost gets the entire dataset to hunt for non-linear interactions
    out_xg, oof_xg, _ = get_prediction_from_model(
        is_detailed=True, to_subtract=False, model_name="xgb"
    )

    print("\nTraining base model #3: XGBoost (binary logistic):")
    # this one directly predicts the probability of win (no margin + calibrator step)
    out_xg_bin, oof_xg_bin, _ = get_prediction_from_model(
        is_detailed=True, to_subtract=False, model_name="xgb_bin"
    )

    # --------------------------------- Ensemble Optimization -------------------------------------------------------- #
    # --------------------------------- Stacking (LogLoss) ----------------------------------------------------------- #
    # in this part, due to us having multiple models, we perform "stacking" to determine the weights of the prediction
    # each model should have based on history:
    print("\nStacking (Meta Logistic) ---")

    # For meta training, we want IsMen aligned with the LR training rows
    # (same tournament rows/order as oof_det / y_targets thanks to index-aligned OOF).
    X_meta_base, _ = get_training_set(is_detailed=True, to_subtract=True)
    is_men_train = X_meta_base["IsMen"].astype(int).values

    # building the meta training features:
    meta_feats = np.column_stack([oof_detailed, oof_xg, oof_xg_bin])

    # Fit two metamodels (Men/Women) on OOF predictions to directly optimize LogLoss:
    meta_men = LogisticRegression(max_iter=2000)
    meta_women = LogisticRegression(max_iter=2000)

    # building the masks for splitting rows to men and women:
    men_mask = (is_men_train == 1)
    women_mask = (is_men_train == 0)

    # fitting models using OOF base predictions:
    meta_men.fit(meta_feats[men_mask], y_targets[men_mask])
    meta_women.fit(meta_feats[women_mask], y_targets[women_mask])

    oof_meta = np.empty_like(y_targets, dtype=float)
    # generate probabilities from the meta models for the same training rows:
    oof_meta[men_mask] = meta_men.predict_proba(meta_feats[men_mask])[:, 1]
    oof_meta[women_mask] = meta_women.predict_proba(meta_feats[women_mask])[:, 1]
    oof_meta = np.clip(oof_meta, 1e-6, 1 - 1e-6)

    print(f">> OOF Meta Brier: {brier_score_loss(y_targets, oof_meta):.5f} <<")
    print(f">> OOF Meta LogLoss: {log_loss(y_targets, oof_meta):.5f} <<")

    # --------------------------------- Apply Meta & Save ------------------------------------------------------------ #
    out = out_detailed.copy()

    # Build meta features for submission
    # ID format is SSSS_XXXX_YYYY and XXXX is the LOWER TeamID (A_TeamID in our parsing)
    season_sub, teamA_sub, teamB_sub = parse_submission_ids(out)
    is_men_sub = (teamA_sub < 3000).astype(int)

    meta_sub = np.column_stack([out_detailed["Pred"].values, out_xg["Pred"].values, out_xg_bin["Pred"].values])

    final_pred = np.empty(len(out), dtype=float)
    men_mask_sub = (is_men_sub == 1)
    wom_mask_sub = (is_men_sub == 0)

    final_pred[men_mask_sub] = meta_men.predict_proba(meta_sub[men_mask_sub])[:, 1]
    final_pred[wom_mask_sub] = meta_women.predict_proba(meta_sub[wom_mask_sub])[:, 1]

    out["Pred"] = np.clip(final_pred, 1e-5, 1 - 1e-5)
    #out["Pred"] = np.where(out["Pred"] >= 0.8, 1.0, out["Pred"])
    #out["Pred"] = np.where(out["Pred"] <= 0.2, 0.0, out["Pred"])

    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
