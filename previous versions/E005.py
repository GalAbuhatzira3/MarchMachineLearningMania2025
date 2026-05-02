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
# regular season results:
MEN_REG = "MRegularSeasonCompactResults.csv"
WOMEN_REG = "WRegularSeasonCompactResults.csv"
DAYS_IN_REGULAR_SEASON = 132

# regular season detailed results:
MEN_REG_DET = "MRegularSeasonDetailedResults.csv"
WOMEN_REG_DET = "WRegularSeasonDetailedResults.csv"

# there is no prior info in the detailed csvs:
MEN_MIN_SEASON = 2003
WOMEN_MIN_SEASON = 2010

# tournament results:
MEN_TOUR = "MNCAATourneyCompactResults.csv"
WOMEN_TOUR = "WNCAATourneyCompactResults.csv"

# the matchups we must predict:
SAMPLE_SUB = "SampleSubmissionStage2.csv"

# tournament seeds:
MEN_SEEDS = "MNCAATourneySeeds.csv"
WOMEN_SEEDS = "WNCAATourneySeeds.csv"

BASE_ELO_SCORE = 1500

# These files contain each team’s tournament seed per season (the committee’s pre-tournament strength ranking).
# This is very predictive for win probability.


# ------------------------------ CSV Handling ------------------------------------------------------------------------ #
def read_csv(name: str, usecols=None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, usecols=usecols)  # usecols loads only columns we need (faster + less memory)


def get_season_elo_ratings(df, k_factor=25, hca=125):
    """
    Calculates dynamic Elo ratings for NCAA basketball teams across multiple seasons.
    Incorporates Home-Court Advantage (HCA), Margin of Victory (MOV), and off-season mean reversion.

    Returns: A DataFrame of end-of-season Elo ratings for every active team per season.
    """
    # 1. SETUP: Elo is sequential, so the data must be perfectly chronological
    df = df.sort_values(by=['Season', 'DayNum']).reset_index(drop=True)
    print(df)

    elo_ratings = {}  # Running dictionary of current team ratings {TeamID: Rating}
    season_results = []  # List of dictionaries to build the final DataFrame
    active_teams = set()  # Tracks teams that played in the current season loop

    current_season = df['Season'].iloc[0] if not df.empty else None

    # 2. MAIN LOOP: Iterate chronologically game by game
    for index, row in df.iterrows():
        season = row['Season']
        w_id, l_id = row['WTeamID'], row['LTeamID']
        w_score, l_score = row['WScore'], row['LScore']
        w_loc = row['WLoc']

        # 3. OFF-SEASON REVERSION: Triggered when the season year rolls over
        if season != current_season:
            # Snapshot the final ratings of the season that just ended
            for team in active_teams:
                season_results.append({
                    'Season': current_season,
                    'TeamID': team,
                    'EloEnd': elo_ratings[team]
                })

            # Regress all ratings 25% toward the league average (1500) for the new season
            # This accounts for roster changes, graduations, and transfers.
            for team in elo_ratings:
                elo_ratings[team] = (0.75 * elo_ratings[team]) + (0.25 * BASE_ELO_SCORE)

            # Reset tracking variables for the fresh season
            current_season = season
            active_teams = set()

        # 4. GAME INITIALIZATION: Give unseen teams the baseline 1500 rating
        if w_id not in elo_ratings: elo_ratings[w_id] = BASE_ELO_SCORE
        if l_id not in elo_ratings: elo_ratings[l_id] = BASE_ELO_SCORE

        # Mark teams as active, so they are included in the end-of-season snapshot
        active_teams.add(w_id)
        active_teams.add(l_id)

        rating_winner = elo_ratings[w_id]
        rating_loser = elo_ratings[l_id]

        # 5. HOME-COURT ADVANTAGE (HCA): Temporarily boost the home team's rating
        if w_loc == 'H':
            rating_winner_adjustment = rating_winner + hca
            rating_loser_adjustment = rating_loser
        elif w_loc == 'A':
            rating_winner_adjustment = rating_winner
            rating_loser_adjustment = rating_loser + hca
        else:  # Neutral location ('N') - no adjustments
            rating_winner_adjustment = rating_winner
            rating_loser_adjustment = rating_loser

        # 6. EXPECTED OUTCOME: Calculate the probability of the winning team winning
        # Uses the logistic distribution based on the rating difference
        e_w = 1 / (1 + 10 ** ((rating_loser_adjustment - rating_winner_adjustment) / 400))

        # 7. MARGIN OF VICTORY (MOV): Scale the rating update based on the blowout factor
        # The denominator ensures favorites aren't over-rewarded for beating weak teams
        mov = w_score - l_score
        delta_r = rating_winner_adjustment - rating_loser_adjustment
        mov_m = ((mov + 3) ** 0.8) / (7.5 + 0.006 * delta_r)

        # 8. RATING UPDATE: Calculate the point shift and apply it to both teams
        # K-factor dictates volatility. (1 - e_w) dictates the surprise factor.
        shift = k_factor * mov_m * (1 - e_w)

        # Zero-sum system: winner gains what the loser drops
        elo_ratings[w_id] += shift
        elo_ratings[l_id] -= shift

    # 9. FINAL SNAPSHOT: The loop ends before saving the very last season, so we save it here
    if current_season is not None:
        for team in active_teams:
            season_results.append({
                'Season': current_season,
                'TeamID': team,
                'EloEnd': elo_ratings[team]
            })

    # 10. FORMATTING: Convert to DataFrame and sort cleanly
    final_df = pd.DataFrame(season_results)
    final_df = final_df.sort_values(
        by=['Season', 'EloEnd'],
        ascending=[True, False]
    ).reset_index(drop=True)

    return final_df


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
    merged = df.merge(featsA, left_on=["Season", "A"], right_on=["Season", "TeamID"], how="left").drop(
        columns=["TeamID"])
    # attach B_... features by matching (Season, B)
    merged = merged.merge(featsB, left_on=["Season", "B"], right_on=["Season", "TeamID"], how="left").drop(
        columns=["TeamID"])
    # Error handling: if a team-season feature is missing for some reason, we use 0.0
    merged = merged.fillna(0.0)

    # E002 CHANGED: added "EloEnd" to base
    # E003 CHANGED: add more detailed rate-based features
    base = [col for col in feats.columns if col not in ["Season", "TeamID", "SeedNum"]]
    X = pd.DataFrame({k: merged[f"A_{k}"] - merged[f"B_{k}"] for k in base})

    # a list of 1s and 0s. y[i] = 1 <-> A won the tournament game that the i'th row in 'merged' represents.
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

    base = [col for col in feats.columns if col not in ["Season", "TeamID", "SeedNum"]]

    X = pd.DataFrame({k: fA[k] - fB[k] for k in base})
    return X


def get_regular(is_detailed: bool):
    """get regular season table (detailed or not), combining regular season results from both men and women"""

    def extract_regular_from_csv(csv):
        # Each row represents one game and has the columns:
        regular_data = read_csv(csv)
        regular_data = regular_data[regular_data["DayNum"] <= DAYS_IN_REGULAR_SEASON]
        return regular_data

    men_regular = extract_regular_from_csv(MEN_REG_DET if is_detailed else MEN_REG)  # men regular season results
    women_regular = extract_regular_from_csv(
        WOMEN_REG_DET if is_detailed else WOMEN_REG)  # women regular season results
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

    # Extract the numeric part (handles play-in like 'X16a' too)
    seed_num = pd.to_numeric(seeds["Seed"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    seeds["SeedNum"] = seed_num.fillna(17).astype(int)

    # Strength-style encoding (bigger = better)
    seeds["SeedStrength"] = 17 - seeds["SeedNum"]

    seeds["HasSeed"] = 1
    seeds = seeds.drop_duplicates(["Season", "TeamID"], keep="last")
    return seeds[["Season", "TeamID", "SeedNum", "SeedStrength", "HasSeed"]]


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
    regular = get_regular(is_detailed=False)
    regular_detailed = get_regular(is_detailed=True)

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
    # diff per game.
    # Theory-wise: we are making the assumption that tournament strength is correlated with regular-season performance,
    # as better teams win more games, score more, concede less (again, just an initial version)
    features = build_team_season_features(regular)

    # E003: add detailed (rate-based) features
    detailed_features = build_team_season_detailed_features(regular_detailed)
    features = features.merge(detailed_features, on=["Season", "TeamID"], how="inner")

    # E002: plug EloEnd into feats (one extra column per (Season, TeamID))
    # In E001, we had feats, which was used to describe each team-season only by:
    #   WinPct, PF_perG, PA_perG, Diff_perG, G
    # Now it also include:
    #   EloEnd = strength rating learned from the sequence of games and opponent quality
    # It should help us predict better since Win% and point diff are raw totals, and don't fully account for:
    #   strength of opponents (easy schedule vs hard schedule)
    #   how informative win/losses are (upset vs expected win)
    # the idea of Elo built exactly around that: beating a strong team gives more credit than beating a weak team.
    elo = get_season_elo_ratings(regular)  # one row per (Season, TeamID) with EloEnd
    features = features.merge(elo, on=["Season", "TeamID"], how="left")  # just a join
    features["EloEnd"] = features["EloEnd"].fillna(1500.0)  # natural average team in Elo scale.

    # E004: add tournament seeding info (per team-season)
    seeds = get_seeds()
    features = features.merge(seeds, on=["Season", "TeamID"], how="inner")
    features["HasSeed"] = features["HasSeed"].fillna(0).astype(int)
    features["SeedNum"] = features["SeedNum"].fillna(17).astype(int)  # worst-case if missing
    features["SeedStrength"] = features["SeedStrength"].fillna(0.0)

    # [2] Build a supervised learning dataset (X,y)
    X, y, groups = build_training_from_tournament(tournament, features)

    print(f"Feature rows (team-season): {len(features):,}")
    print(f"Training rows: {len(X):,} | Positive rate: {y.mean():.3f}")

    # --------------------------------- Building the Model ----------------------------------------------------------- #
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])

    # --------------------------------- Validating with season-based CV (brier score) -------------------------------- #
    folds = 5
    gkf = GroupKFold(n_splits=folds)
    briers = []
    for i, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]  # [P[A loses], P[A wins]] , we need P[A wins]
        b = brier_score_loss(y[va], p)
        briers.append(b)
        print(f"Fold {i}/{folds}: Brier={b:.5f}")
    print(f"CV Brier mean={np.mean(briers):.5f} | std={np.std(briers):.5f}")

    # --------------------------------- Train on all history --------------------------------------------------------- #
    model.fit(X, y)

    # --------------------------------- Submission features (Actual prediction) -------------------------------------- #
    season, a, b = parse_submission_ids(sub)
    X_sub = build_matchup_features(season, a, b, features)

    pred = model.predict_proba(X_sub)[:, 1]
    pred = np.clip(pred, 1e-5, 1 - 1e-5)

    out = pd.DataFrame({"ID": sub["ID"], "Pred": pred})
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    # E002 idea: EloEnd helps because it captures things that E001 didn't:
    #   Strength of schedule: a team going 20-10 vs strong teams, can be better that team going 25-5 vs weak team.
    #   Value of wins: beating a strong opponent gives a bigger rating increase than beating a weak one.
    #   It compresses a lot of season information into one number that correlates strongly with winning.
    main()
