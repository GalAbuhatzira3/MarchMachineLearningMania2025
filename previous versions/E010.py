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

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ------------------------------------------------- Paths + file names ----------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Auto-detect project paths
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "submission.csv"  # output file name and path

# ------------------------------------------------- Specific CSVs ---------------------------------------------------- #
# regular season compact results:
MEN_REG = "MRegularSeasonCompactResults.csv"
WOMEN_REG = "WRegularSeasonCompactResults.csv"
DAYS_IN_REGULAR_SEASON = 132

# regular season detailed results:
MEN_REG_DET = "MRegularSeasonDetailedResults.csv"
WOMEN_REG_DET = "WRegularSeasonDetailedResults.csv"

# tournament compact results:
MEN_TOUR = "MNCAATourneyCompactResults.csv"
WOMEN_TOUR = "WNCAATourneyCompactResults.csv"

# tournament detailed results:
MEN_TOUR_DET = "MNCAATourneyDetailedResults.csv"
WOMEN_TOUR_DET = "WNCAATourneyDetailedResults.csv"

# the matchups we must predict:
SAMPLE_SUB = "SampleSubmissionStage2.csv"

# tournament seeds:
MEN_SEEDS = "MNCAATourneySeeds.csv"
WOMEN_SEEDS = "WNCAATourneySeeds.csv"

BASE_ELO_SCORE = 1500


# These files contain each team’s tournament seed per season (the committee’s pre-tournament strength ranking).
# This is very predictive for win probability.


# ------------------------------ Loading Data ------------------------------------------------------------------------ #
def read_csv(name: str, usecols=None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, usecols=usecols)  # usecols loads only columns we need (faster + less memory)


def get_regular(is_detailed: bool):
    """get regular season table (detailed or not), combining regular season results from both men and women"""

    def extract_regular_from_csv(csv):
        # Each row represents one game and has the columns:
        regular_data = read_csv(csv)
        regular_data = regular_data[regular_data["DayNum"] <= DAYS_IN_REGULAR_SEASON]
        return regular_data

    men_regular = extract_regular_from_csv(MEN_REG_DET if is_detailed else MEN_REG)
    women_regular = extract_regular_from_csv(WOMEN_REG_DET if is_detailed else WOMEN_REG)
    # Concatenate to one table:
    regular = pd.concat([men_regular, women_regular], ignore_index=True)
    print(f"Regular season loaded! Loaded games: {len(regular):,}")
    return regular


def get_tournament(is_detailed: bool):
    # Each row is a real tournament game, and has the columns:
    tournament_columns = ["Season", "WTeamID", "LTeamID"]
    men_tournament = read_csv(MEN_TOUR_DET if is_detailed else MEN_TOUR, usecols=tournament_columns)
    women_tournament = read_csv(WOMEN_TOUR_DET if is_detailed else WOMEN_TOUR, usecols=tournament_columns)
    # Concatenate to one table:
    tournament = pd.concat([men_tournament, women_tournament], ignore_index=True)
    print(f"Tournament loaded! Loaded games: {len(tournament):,}")
    return tournament


# ---------------------------- Training ------------------------------------------------------------------------------ #
def build_team_season_compact_features(regular: pd.DataFrame) -> pd.DataFrame:
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
    loser["G"] = 1
    loser["W"] = 0
    # Concatenation of winners and losers:
    games = pd.concat([winner, loser], ignore_index=True)
    # Building the feats table as grouping over the key (Season, TeamID) because we want the stats of the team
    # for every team and every season
    feats = games.groupby(["Season", "TeamID"], as_index=False).agg(
        G=("G", "sum"),  # summing gives us total games TeamID played in Season
        W=("W", "sum"),  # summing gives us total wins TeamID won in Season
        PF=("PF", "sum"),  # summing gives us total points TeamID scored in Season
        PA=("PA", "sum"),  # summing gives us total points TeamID conceded in Season
    )
    # Convert totals into rates:
    feats["WinPct"] = feats["W"] / feats["G"]  # winning percentage
    feats["PF_perG"] = feats["PF"] / feats["G"]  # points scored per game
    feats["PA_perG"] = feats["PA"] / feats["G"]  # points conceded per game
    feats["Diff_perG"] = (feats["PF"] - feats["PA"]) / feats["G"]  # scoring margin per game
    # Return only the relevant columns: (without the ones used for the calculations)
    return feats[["Season", "TeamID", "WinPct", "PF_perG", "PA_perG", "Diff_perG", "G"]]


def build_team_season_detailed_features(regular_detailed: pd.DataFrame) -> pd.DataFrame:
    # the function converts the detailed game table into a (Season, TeamID) features table.
    # the compact CSV only has scores + winners/losers. The detailed one has box score stats, which builds richer
    # "team strength" features

    # Helper with division by zero:
    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return (num / den).fillna(0.0)

    # Many advanced stats work best per possession, not per game, because pace varies a lot. This is a formula to
    # estimate team's possession based of its field goal attempts, offensive rebounds, turnovers and free throw attempts
    def get_possession(FGA, OR, TO, FTA):
        return (FGA + TO + 0.44 * FTA) - OR

    def get_winner_or_loser(pd, is_winner):
        id_char = 'W' if is_winner else 'L'
        opp_char = 'L' if is_winner else 'W'
        return pd.DataFrame({
            "Season": df["Season"],
            "TeamID": df[f"{id_char}TeamID"],
            "G": 1,
            "PF": df[f"{id_char}Score"],  # points scored
            "PA": df[f"{opp_char}Score"],  # points conceded

            # offense box score
            "FGM": df[f"{id_char}FGM"], "FGA": df[f"{id_char}FGA"],
            "FGM3": df[f"{id_char}FGM3"], "FGA3": df[f"{id_char}FGA3"],
            "FTM": df[f"{id_char}FTM"], "FTA": df[f"{id_char}FTA"],
            "OR": df[f"{id_char}OR"], "DR": df[f"{id_char}DR"],
            "Ast": df[f"{id_char}Ast"], "TO": df[f"{id_char}TO"],
            "Stl": df[f"{id_char}Stl"], "Blk": df[f"{id_char}Blk"], "PFoul": df[f"{id_char}PF"],

            # pace denominators
            "Poss": df[f"{id_char}_Poss"], "OppPoss": df[f"{opp_char}_Poss"],

            # opponent stats (for defensive allowed features / reb denominators)
            "OppFGM": df[f"{opp_char}FGM"], "OppFGA": df[f"{opp_char}FGA"],
            "OppFGM3": df[f"{opp_char}FGM3"], "OppFGA3": df[f"{opp_char}FGA3"],
            "OppFTM": df[f"{opp_char}FTM"], "OppFTA": df[f"{opp_char}FTA"],
            "OppOR": df[f"{opp_char}OR"], "OppDR": df[f"{opp_char}DR"],
            "OppTO": df[f"{opp_char}TO"],
        })

    df = regular_detailed.copy()

    # Possessions approximation:
    df["W_Poss"] = get_possession(df["WFGA"], df["WOR"], df["WTO"], df["WFTA"])  # winner
    df["L_Poss"] = get_possession(df["LFGA"], df["LOR"], df["LTO"], df["LFTA"])  # loser

    # Converting each game into two "team rows" (winner, loser):
    winner = get_winner_or_loser(pd, is_winner=1)
    loser = get_winner_or_loser(pd, is_winner=0)
    games = pd.concat([winner, loser], ignore_index=True)
    # Now every row in games means: in this Season game, this TeamID had these box score totals and these opponent
    # totals. Now all there is left to do is to group by (Season, TeamID)
    totals = games.groupby(["Season", "TeamID"], as_index=False).sum(numeric_only=True)

    # Four Factors / efficiency style features (from season totals)
    totals["eFG"] = safe_div(totals["FGM"] + 0.5 * totals["FGM3"], totals["FGA"])  # shooting efficiency
    totals["FTR"] = safe_div(totals["FTA"], totals["FGA"])  # how often do this team gets to the line
    totals["TOV"] = safe_div(totals["TO"], totals["Poss"])  # turnovers per possession
    totals["ORB"] = safe_div(totals["OR"], totals["OR"] + totals["OppDR"])  # off reb %
    totals["DRB"] = safe_div(totals["DR"], totals["DR"] + totals["OppOR"])  # def reb %
    totals["Pace"] = safe_div(totals["Poss"], totals["G"])  # possessions per game
    totals["OffEff"] = safe_div(totals["PF"], totals["Poss"])  # points scored per possession
    totals["DefEff"] = safe_div(totals["PA"], totals["OppPoss"])  # points allowed per possession
    totals["NetEff"] = totals["OffEff"] - totals["DefEff"]

    # Explicit 3PT reliance + accuracy
    totals["3PAr"] = safe_div(totals["FGA3"], totals["FGA"])  # how much we rely on 3s
    totals["3P%"] = safe_div(totals["FGM3"], totals["FGA3"])  # how well we hit 3s
    totals["FT%"] = safe_div(totals["FTM"], totals["FTA"])  # free throw accuracy

    # True Shooting %
    totals["TS"] = safe_div(totals["PF"], 2.0 * (totals["FGA"] + 0.44 * totals["FTA"]))

    # Rates
    totals["AST_Rate"] = safe_div(totals["Ast"], totals["Poss"])  # assists rates (sharing the ball)
    totals["STL_Rate"] = safe_div(totals["Stl"], totals["OppPoss"])  # steals rates (extra possession)
    totals["BLK_Rate"] = safe_div(totals["Blk"], totals["OppFGA"])  # block rates (defence indicator)
    totals["PF_Rate"] = safe_div(totals["PFoul"], totals["Poss"])  # personal fouls rate (bad defence indicator)

    # Defensive allowed profile
    totals["Def_eFG"] = safe_div(totals["OppFGM"] + 0.5 * totals["OppFGM3"], totals["OppFGA"])
    totals["Def_3PAr"] = safe_div(totals["OppFGA3"], totals["OppFGA"])
    totals["Def_3P%"] = safe_div(totals["OppFGM3"], totals["OppFGA3"])
    totals["Def_FTR"] = safe_div(totals["OppFTA"], totals["OppFGA"])

    detailed = totals[[
        "Season", "TeamID",
        "eFG", "FTR", "TOV", "ORB", "DRB", "Pace", "NetEff",
        "3PAr", "3P%", "FT%", "TS",
        "AST_Rate", "STL_Rate", "BLK_Rate", "PF_Rate",
        "Def_eFG", "Def_3PAr", "Def_3P%", "Def_FTR",
    ]].copy()

    return detailed


def build_elo_end_features(regular: pd.DataFrame, k: float = 20.0, base: float = 1500.0, reversion: float = 0.75,
                           hca: float = 75.0, momentum_day: int = 100) -> pd.DataFrame:
    games = regular[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"]].copy()
    games = games.sort_values(["Season", "DayNum"], kind="mergesort")

    def expected(a_rating: float, b_rating: float) -> float:
        return 1.0 / (1.0 + 10 ** ((b_rating - a_rating) / 400.0))

    rows = []
    current_season = None
    ratings: dict[int, float] = {}

    # --- NEW: Momentum Trackers ---
    ratings_snapshot: dict[int, float] = {}
    snapshot_taken = False

    for season, day, winner, loser, w_score, l_score, w_loc in games.itertuples(index=False):
        # On new season: save previous season ratings and apply mean reversion
        if current_season is None or season != current_season:
            if current_season is not None:
                for tid, r in ratings.items():
                    # Calculate how much Elo they gained/lost since Day 100
                    r_past = ratings_snapshot.get(tid, base)
                    elo_delta = r - r_past
                    rows.append((current_season, tid, r, elo_delta))

                # Season-to-Season Carryover
                for tid in ratings:
                    ratings[tid] = (reversion * ratings[tid]) + ((1.0 - reversion) * base)

            current_season = season

            # Reset the momentum trackers for the new season
            ratings_snapshot = {}
            snapshot_taken = False

        # --- NEW: Take the snapshot the moment we reach Day 100 ---
        if not snapshot_taken and day >= momentum_day:
            ratings_snapshot = ratings.copy()
            snapshot_taken = True

        winner_rating = ratings.get(winner, base)
        loser_rating = ratings.get(loser, base)

        winner_rating_adj = winner_rating
        loser_rating_adj = loser_rating

        if w_loc == 'H':
            winner_rating_adj += hca
        elif w_loc == 'A':
            loser_rating_adj += hca

        expected_winner = expected(winner_rating_adj, loser_rating_adj)
        expected_loser = 1.0 - expected_winner

        # Logarithmic Margin of Victory (MOV)
        mov = w_score - l_score
        mov_multiplier = np.log(mov) + 1.0

        dynamic_k = k * mov_multiplier

        ratings[winner] = winner_rating + dynamic_k * (1.0 - expected_winner)
        ratings[loser] = loser_rating + dynamic_k * (0.0 - expected_loser)

    # Save last season
    if current_season is not None:
        for tid, r in ratings.items():
            r_past = ratings_snapshot.get(tid, base)
            elo_delta = r - r_past
            rows.append((current_season, tid, r, elo_delta))

    # We now output both EloEnd and EloDelta
    elo = pd.DataFrame(rows, columns=["Season", "TeamID", "EloEnd", "EloDelta"])
    elo = elo.sort_values(["Season", "TeamID"]).drop_duplicates(["Season", "TeamID"], keep="last")

    return elo


def get_seeds() -> pd.DataFrame:
    # Loads tournament seeds (men + women) and converts the raw 'Seed' string in the csv (e.g. 'W01', 'X16a')
    # into numeric features per (Season, TeamID).

    # We use:
    # - SeedNum: 1..16  (lower is better)
    # - SeedStrength: 17 - SeedNum  (higher is better, so it matches our "positive = stronger")
    # - HasSeed: 1 if we have a seed for that team-season else 0
    usecols = ["Season", "Seed", "TeamID"]
    men = read_csv(MEN_SEEDS, usecols=usecols)
    women = read_csv(WOMEN_SEEDS, usecols=usecols)
    # concatenating men and women seeds, into one unified (Season, TeamID) table for both genders:
    seeds = pd.concat([men, women], ignore_index=True)

    # Extract the numeric part
    seed_num = pd.to_numeric(seeds["Seed"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    seeds["SeedNum"] = seed_num.fillna(17).astype(int)

    # 1. Standard linear strength (bigger = better).
    # Log-transformed encoding for linear models
    seeds["SeedLog"] = np.log(seeds["SeedNum"])

    seeds = seeds.drop_duplicates(["Season", "TeamID"], keep="last")

    # Notice we removed SeedStrength and added SeedLog here
    return seeds[["Season", "TeamID", "SeedNum", "SeedLog"]]




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
    mrg = df.merge(featsA, left_on=["Season", "A"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])
    # attach B_... features by matching (Season, B)
    mrg = mrg.merge(featsB, left_on=["Season", "B"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])

    base = [col for col in feats.columns if col not in ["Season", "TeamID", "SeedNum"]]
    X = pd.DataFrame({k: mrg[f"A_{k}"] - mrg[f"B_{k}"] for k in base})

    # a list of 1s and 0s. y[i] = 1 <-> A won the tournament game that the i'th row in 'merged' represents.
    y = mrg["y"].to_numpy()
    groups = mrg["Season"].to_numpy()
    return X, y, groups


def build_matchup_features(season: pd.Series, a: pd.Series, b: pd.Series, feats: pd.DataFrame) -> pd.DataFrame:
    keysA = pd.DataFrame({"Season": season, "TeamID": a})
    keysB = pd.DataFrame({"Season": season, "TeamID": b})

    fA = keysA.merge(feats, on=["Season", "TeamID"], how="left").fillna(0.0)
    fB = keysB.merge(feats, on=["Season", "TeamID"], how="left").fillna(0.0)

    base = [col for col in feats.columns if col not in ["Season", "TeamID", "SeedNum"]]

    X = pd.DataFrame({k: fA[k] - fB[k] for k in base})
    return X


def parse_submission_ids(sub: pd.DataFrame):
    parts = sub["ID"].str.split("_", expand=True)
    season = parts[0].astype(int)
    a = parts[1].astype(int)
    b = parts[2].astype(int)
    return season, a, b


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
    regular_compact = get_regular(is_detailed=False)  # Men: 1985 - 2025, Women: 1998 - 2025
    print(regular_compact)
    regular_detailed = get_regular(is_detailed=True)  # Men: 2003 - 2025, Women: 2010 - 2025
    print(regular_detailed)

    # (B) Tournament games (men + women)
    #   Purpose: This will be our training set (labels) - who actually won in the tournament
    tournament_compact = get_tournament(is_detailed=False)
    print(tournament_compact)
    tournament_detailed = get_tournament(is_detailed=True)
    print(tournament_detailed)

    # (C) The matchups we must predict
    # SampleSubmissionStage2.csv (131,407 rows) contains ID like 2025_1101_1102,  2025_1101_1103
    #   Purpose: this is the list of all matchups Kaggle expects in our submission.
    #            We must output a probability for each.
    sub = read_csv(SAMPLE_SUB, usecols=["ID", "Pred"])
    print(f"Loaded submission rows: {len(sub):,}")

    # --------------------------------- Training --------------------------------------------------------------------- #
    # [1] Turn regular-season games into team season stats:
    # Evaluating Elo:
    elo = build_elo_end_features(regular_compact)
    # Evaluating Seeds:
    seeds = get_seeds()
    # getting features and merging with Elo and seeds:
    features_compact = build_team_season_compact_features(regular_compact)
    print("The compact features table: ")
    print(features_compact)
    features_compact = features_compact.merge(elo, on=["Season", "TeamID"], how="left")
    features_compact = features_compact.merge(seeds, on=["Season", "TeamID"], how="left")

    # --- NEW: Inject Recent Form (Last 30 Days) ---
    features_compact["SeedNum"] = features_compact["SeedNum"].fillna(17)
    features_compact["SeedLog"] = features_compact["SeedLog"].fillna(np.log(17))
    print("The compact features table (with Elo and Seeds): ")
    print(features_compact)

    # E003: add detailed (rate-based) features
    features_detailed = build_team_season_detailed_features(regular_detailed)
    features_detailed = features_detailed.merge(features_compact, on=["Season", "TeamID"], how="inner")
    print("The detailed features table: ")
    print(features_detailed)

    # [2] Build supervised learning datasets (X,y)
    X_comp, y_comp, groups_comp = build_training_from_tournament(tournament_compact, features_compact)
    X_det, y_det, groups_det = build_training_from_tournament(tournament_detailed, features_detailed)

    print(f"Compact Training rows:  {len(X_comp):,}")
    print(X_comp)
    print(f"Detailed Training rows: {len(X_det):,}")
    print(X_det)

    # --------------------------------- Building the Models ---------------------------------------------------------- #
    model_compact = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    model_detailed = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])

    # --------------------------------- Train on all history & Predict ----------------------------------------------- #
    model_compact.fit(X_comp, y_comp)
    model_detailed.fit(X_det, y_det)

    season, a, b = parse_submission_ids(sub)
    X_sub_comp = build_matchup_features(season, a, b, features_compact)
    X_sub_det = build_matchup_features(season, a, b, features_detailed)

    # Get predictions from both models
    pred_comp = model_compact.predict_proba(X_sub_comp)[:, 1]
    pred_det = model_detailed.predict_proba(X_sub_det)[:, 1]

    # --- THE ENSEMBLE ---
    # We average the predictions.
    # If you later decide the Compact model is harming you, change this to: final_pred = pred_det
    final_pred = 0.7*pred_comp + 0.3*pred_det
    final_pred = np.clip(final_pred, 1e-5, 1 - 1e-5)

    out = pd.DataFrame({"ID": sub["ID"], "Pred": final_pred})
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    # E002 idea: EloEnd helps because it captures things that E001 didn't:
    #   Strength of schedule: a team going 20-10 vs strong teams, can be better that team going 25-5 vs weak team.
    #   Value of wins: beating a strong opponent gives a bigger rating increase than beating a weak one.
    #   It compresses a lot of season information into one number that correlates strongly with winning.
    main()
