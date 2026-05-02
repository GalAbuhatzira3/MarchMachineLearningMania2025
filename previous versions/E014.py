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
from sklearn.metrics import brier_score_loss

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


def get_regular(is_detailed: bool):
    men_regular = read_csv(MEN_REG_DET if is_detailed else MEN_REG)
    women_regular = read_csv(WOMEN_REG_DET if is_detailed else WOMEN_REG)
    regular = pd.concat([men_regular, women_regular], ignore_index=True)
    return regular  # compact: 329,928 rows, detailed: 200,590 rows


def get_tournament(is_detailed: bool):
    tournament_columns = ["Season", "WTeamID", "LTeamID"]
    men_tournament = read_csv(MEN_TOUR_DET if is_detailed else MEN_TOUR, usecols=tournament_columns)
    women_tournament = read_csv(WOMEN_TOUR_DET if is_detailed else WOMEN_TOUR, usecols=tournament_columns)
    tournament = pd.concat([men_tournament, women_tournament], ignore_index=True)
    return tournament  # compact: 4168 rows, detailed: 2276 rows


# ---------------------------- Training ------------------------------------------------------------------------------ #
def build_team_season_compact_features(regular: pd.DataFrame) -> pd.DataFrame:
    winner = regular[["Season", "WTeamID", "WScore", "LScore"]].copy()
    winner.columns = ["Season", "TeamID", "PF", "PA"]
    winner["G"] = 1
    winner["W"] = 1

    loser = regular[["Season", "LTeamID", "LScore", "WScore"]].copy()
    loser.columns = ["Season", "TeamID", "PF", "PA"]
    loser["G"] = 1
    loser["W"] = 0

    games = pd.concat([winner, loser], ignore_index=True)

    feats = games.groupby(["Season", "TeamID"], as_index=False).agg(
        G=("G", "sum"),
        W=("W", "sum"),
        PF=("PF", "sum"),
        PA=("PA", "sum"),
    )

    feats["WinPct"] = feats["W"] / feats["G"]
    feats["PF_perG"] = feats["PF"] / feats["G"]
    feats["PA_perG"] = feats["PA"] / feats["G"]
    feats["Diff_perG"] = (feats["PF"] - feats["PA"]) / feats["G"]

    return feats[["Season", "TeamID", "WinPct", "PF_perG", "PA_perG", "Diff_perG", "G"]]


def build_team_season_detailed_features(regular_detailed: pd.DataFrame) -> pd.DataFrame:
    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return (num / den).fillna(0.0)

    def get_possession(FGA, OR, TO, FTA):
        return (FGA + TO + 0.44 * FTA) - OR

    def get_winner_or_loser(pd, is_winner):
        id_char = 'W' if is_winner else 'L'
        opp_char = 'L' if is_winner else 'W'

        temp_df = pd.DataFrame({
            "Season": df["Season"],
            "TeamID": df[f"{id_char}TeamID"],
            "G": 1,
            "NumOT": df["NumOT"],  # Fetch NumOT for standardization

            "PF": df[f"{id_char}Score"],
            "PA": df[f"{opp_char}Score"],
            "FGM": df[f"{id_char}FGM"], "FGA": df[f"{id_char}FGA"],
            "FGM3": df[f"{id_char}FGM3"], "FGA3": df[f"{id_char}FGA3"],
            "FTM": df[f"{id_char}FTM"], "FTA": df[f"{id_char}FTA"],
            "OR": df[f"{id_char}OR"], "DR": df[f"{id_char}DR"],
            "Ast": df[f"{id_char}Ast"], "TO": df[f"{id_char}TO"],
            "Stl": df[f"{id_char}Stl"], "Blk": df[f"{id_char}Blk"], "PFoul": df[f"{id_char}PF"],

            "OppFGM": df[f"{opp_char}FGM"], "OppFGA": df[f"{opp_char}FGA"],
            "OppFGM3": df[f"{opp_char}FGM3"], "OppFGA3": df[f"{opp_char}FGA3"],
            "OppFTM": df[f"{opp_char}FTM"], "OppFTA": df[f"{opp_char}FTA"],
            "OppOR": df[f"{opp_char}OR"], "OppDR": df[f"{opp_char}DR"],
            "OppTO": df[f"{opp_char}TO"],
        })

        # --- THE OVERTIME FIX ---
        adj_ot = (40.0 + 5.0 * temp_df["NumOT"]) / 40.0
        cols_to_adjust = [
            "PF", "PA", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PFoul",
            "OppFGM", "OppFGA", "OppFGM3", "OppFGA3", "OppFTM", "OppFTA", "OppOR", "OppDR", "OppTO"
        ]

        for col in cols_to_adjust:
            temp_df[col] = temp_df[col] / adj_ot

        temp_df["Poss"] = get_possession(temp_df["FGA"], temp_df["OR"], temp_df["TO"], temp_df["FTA"])
        temp_df["OppPoss"] = get_possession(temp_df["OppFGA"], temp_df["OppOR"], temp_df["OppTO"], temp_df["OppFTA"])

        return temp_df.drop(columns=["NumOT"])

    df = regular_detailed.copy()
    winner = get_winner_or_loser(pd, is_winner=1)
    loser = get_winner_or_loser(pd, is_winner=0)
    games = pd.concat([winner, loser], ignore_index=True)

    totals = games.groupby(["Season", "TeamID"], as_index=False).sum(numeric_only=True)

    totals["OffEff"] = safe_div(totals["PF"], totals["Poss"])
    totals["DefEff"] = safe_div(totals["PA"], totals["OppPoss"])
    totals["NetEff"] = totals["OffEff"] - totals["DefEff"]
    totals["Pace"] = safe_div(totals["Poss"], totals["G"])

    totals["eFG"] = safe_div(totals["FGM"] + 0.5 * totals["FGM3"], totals["FGA"])
    totals["TOV"] = safe_div(totals["TO"], totals["Poss"])
    totals["ORB"] = safe_div(totals["OR"], totals["OR"] + totals["OppDR"])
    totals["FTR"] = safe_div(totals["FTA"], totals["FGA"])

    totals["3PAr"] = safe_div(totals["FGA3"], totals["FGA"])
    totals["3P%"] = safe_div(totals["FGM3"], totals["FGA3"])
    totals["FT%"] = safe_div(totals["FTM"], totals["FTA"])
    totals["TS"] = safe_div(totals["PF"], 2.0 * (totals["FGA"] + 0.44 * totals["FTA"]))

    totals["AST_Rate"] = safe_div(totals["Ast"], totals["Poss"])
    totals["STL_Rate"] = safe_div(totals["Stl"], totals["OppPoss"])
    totals["BLK_Rate"] = safe_div(totals["Blk"], totals["OppFGA"])
    totals["PF_Rate"] = safe_div(totals["PFoul"], totals["Poss"])

    totals["Def_eFG"] = safe_div(totals["OppFGM"] + 0.5 * totals["OppFGM3"], totals["OppFGA"])
    totals["Def_3PAr"] = safe_div(totals["OppFGA3"], totals["OppFGA"])
    totals["Def_3P%"] = safe_div(totals["OppFGM3"], totals["OppFGA3"])
    totals["Def_FTR"] = safe_div(totals["OppFTA"], totals["OppFGA"])

    detailed = totals[[
        "Season", "TeamID",
        "eFG", "FTR", "TOV", "ORB", "Pace", "NetEff",
        "3PAr", "3P%", "FT%", "TS",
        "AST_Rate", "STL_Rate", "BLK_Rate", "PF_Rate",
        "Def_eFG", "Def_3PAr", "Def_3P%", "Def_FTR",
    ]].copy()

    return detailed


def build_elo(
        regular,
        K=20,
        base=1500,
        new_team_base=1300,
        reversion=0.3,
        hca=75,
        early_season_games=20,
        early_k_boost=0.5
) -> pd.DataFrame:
    games = regular.copy().drop(columns="NumOT")
    games = games.sort_values(["Season", "DayNum"], kind="mergesort")
    team_confs = get_conference()
    # Conference map: (Season, TeamID) -> ConfAbbrev
    conf_map = {(int(s), int(t)): c for s, t, c in team_confs[["Season", "TeamID", "ConfAbbrev"]].itertuples(index=False)}

    def expected_win_prop(rating_A, rating_B) -> float:
        """The expected score / win probability of Team A (rating_A) against Team B (rating_B) """
        return 1.0 / (1 + math.pow(10, (rating_B - rating_A) / 400.0))

    def k_multiplier(games_played):
        # 1.5*K at game 0 -> 1*K at game 20
        g = min(games_played, early_season_games)
        return (1.0 + early_k_boost) - early_k_boost * (g/early_season_games)

    rows = []
    seasons = games["Season"].unique()
    prev_elo = {}  # (TeamID, rating) dict of last season
    for season in seasons:  # for every season 1985 until 2024:
        season_games = games[games["Season"] == season]  # take just the games of this season
        active_teams = set(season_games["WTeamID"]).union(set(season_games["LTeamID"]))  # only the teams that played this season
        ratings = {}  # (TeamID, rating) dict of current year
        if not prev_elo:  # this is the first season (1985):
            for teamID in active_teams: ratings[teamID] = base  # basic elo rating since this is the 1st season
        else:  # not the first season, we have data from previous season:
            # the Elo score of the teams this season, will be the mean of the elo score of the teams in their conference
            # last season:
            score = {}
            cnt = {}
            for teamID, rating in prev_elo.items():  # for every team and their rating from last season:
                conf = conf_map.get(int(season) - 1, teamID)
                score[conf] = score.get(conf, 0) + rating
                cnt[conf] = cnt.get(conf, 0) + 1
            prev_conf_mean_elo = {c: score[c] / cnt[c] for c in score if cnt[c] > 0}
            # APPLYING: Year-to-year carryover / mean reversion:
            for teamID in active_teams:  # for every team that participates this season:
                if teamID in prev_elo: # the team also played last year:
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


def get_seeds() -> pd.DataFrame:
    men = read_csv(MEN_SEEDS)  # 1998 - 2025
    women = read_csv(WOMEN_SEEDS)  # 1998 - 2025
    seeds = pd.concat([men, women], ignore_index=True)

    seed_num = pd.to_numeric(seeds["Seed"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    seeds["SeedNum"] = seed_num.fillna(17).astype(int)
    seeds["SeedLog"] = np.log(seeds["SeedNum"])

    seeds = seeds.drop_duplicates(["Season", "TeamID"], keep="last")
    return seeds[["Season", "TeamID", "SeedNum", "SeedLog"]]


def get_massey() -> pd.DataFrame:
    massey = read_csv(MASSEY_ORDINALS)
    massey = massey[massey["RankingDayNum"] == 133].copy()
    massey = massey[massey["SystemName"].isin(["POM"])]
    consensus = massey.groupby(["Season", "TeamID"], as_index=False).agg(MedianRank=("OrdinalRank", "median"))
    # Custom Log Curve
    consensus["RankLog"] = 100 - 4 * np.log(consensus["MedianRank"] + 1) - consensus["MedianRank"] / 22
    return consensus[["Season", "TeamID", "RankLog"]]


# ---------------------------- Building supervised learning dataset (X,y) -------------------------------------------- #
def build_training_from_tournament(tournament: pd.DataFrame, feats: pd.DataFrame):
    df = tournament.copy()
    df["A"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["B"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    df["y"] = (df["A"] == df["WTeamID"]).astype(int)

    featsA = feats.rename(columns={c: f"A_{c}" for c in feats.columns if c not in ["Season", "TeamID"]})
    featsB = feats.rename(columns={c: f"B_{c}" for c in feats.columns if c not in ["Season", "TeamID"]})

    mrg = df.merge(featsA, left_on=["Season", "A"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])
    mrg = mrg.merge(featsB, left_on=["Season", "B"], right_on=["Season", "TeamID"], how="left").drop(columns=["TeamID"])

    mrg = mrg.fillna(0.0)

    # Difference Calculation
    base = [col for col in feats.columns if col not in ["Season", "TeamID", "SeedNum"]]
    X = pd.DataFrame({k: mrg[f"A_{k}"] - mrg[f"B_{k}"] for k in base})

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
    # --------------------------------- Loading Data ----------------------------------------------------------------- #
    regular_compact = get_regular(is_detailed=False)
    regular_detailed = get_regular(is_detailed=True)
    tournament_compact = get_tournament(is_detailed=False)
    tournament_detailed = get_tournament(is_detailed=True)
    sub = read_csv(SAMPLE_SUB, usecols=["ID", "Pred"])

    # --------------------------------- Feature Building ------------------------------------------------------------- #
    elo = build_elo(regular_compact)
    seeds = get_seeds()
    massey = get_massey()

    # Base Compact Matrix
    features_compact = build_team_season_compact_features(regular_compact)
    features_compact = features_compact.merge(elo, on=["Season", "TeamID"], how="left")
    features_compact = features_compact.merge(seeds, on=["Season", "TeamID"], how="left")

    features_compact["SeedNum"] = features_compact["SeedNum"].fillna(17)
    features_compact["SeedLog"] = features_compact["SeedLog"].fillna(np.log(17))

    # Detailed Matrix
    features_detailed = build_team_season_detailed_features(regular_detailed)
    features_detailed = features_detailed.merge(features_compact, on=["Season", "TeamID"], how="inner")

    # --- UNIFIED 2-MODEL ARCHITECTURE (Men & Women Combined) ---
    features_detailed = features_detailed.merge(massey, on=["Season", "TeamID"], how="left")
    print(features_detailed)

    # Impute missing KenPom ranks (All women, some early men) with Rank 175
    # This evaluates to a difference of 0 in the model, safely "turning off" the feature without breaking coefficients!
    features_detailed["RankLog"] = features_detailed["RankLog"].fillna(100 - 4 * np.log(175.0 + 1) - 175.0 / 22)

    # --------------------------------- Build Supervised Datasets ---------------------------------------------------- #
    X_comp, y_comp, groups_comp = build_training_from_tournament(tournament_compact, features_compact)
    X_det, y_det, groups_det = build_training_from_tournament(tournament_detailed, features_detailed)

    print(f"\nTraining Rows - Compact: {len(X_comp):,}, Detailed: {len(X_det):,}")

    # --------------------------------- Building the Models ---------------------------------------------------------- #
    # Pure Logistic Regression
    model_compact = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    model_detailed = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])

    # --------------------------------- CV & Grid Search ------------------------------------------------------------- #
    print("\n--- Running Cross-Validation & Blend Optimization ---")
    folds = 5
    gkf = GroupKFold(n_splits=folds)

    oof_comp = np.zeros(len(X_det))
    oof_det = np.zeros(len(X_det))

    for i, (tr_idx, va_idx) in enumerate(gkf.split(X_det, y_det, groups=groups_det), 1):
        val_seasons = set(groups_det[va_idx])
        print(val_seasons)
        tr_mask_comp = ~np.isin(groups_comp, list(val_seasons))

        model_compact.fit(X_comp.iloc[tr_mask_comp], y_comp[tr_mask_comp])
        X_val_comp_format = X_det.iloc[va_idx][X_comp.columns]
        oof_comp[va_idx] = model_compact.predict_proba(X_val_comp_format)[:, 1]

        model_detailed.fit(X_det.iloc[tr_idx], y_det[tr_idx])
        oof_det[va_idx] = model_detailed.predict_proba(X_det.iloc[va_idx])[:, 1]

    best_weight = 0.5
    best_brier = 1.0

    print("\n--- Ensemble Blend Grid Search ---")
    for w in np.linspace(0.0, 1.0, 21):
        blend_pred = (w * oof_comp) + ((1 - w) * oof_det)
        b = brier_score_loss(y_det, blend_pred)

        if b < best_brier:
            best_brier = b
            best_weight = w

        if w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            print(f"Weight Compact={w:.2f}, Detailed={1 - w:.2f} --> Brier: {b:.5f}")

    print(f"\n>> BEST BLEND: {best_weight:.2f} Compact / {1 - best_weight:.2f} Detailed (Brier: {best_brier:.5f}) <<")

    # --------------------------------- Final Training & Prediction -------------------------------------------------- #
    model_compact.fit(X_comp, y_comp)
    model_detailed.fit(X_det, y_det)

    season, a, b = parse_submission_ids(sub)

    X_sub_comp = build_matchup_features(season, a, b, features_compact)
    X_sub_det = build_matchup_features(season, a, b, features_detailed)

    # Calculate both predictions
    pred_comp = model_compact.predict_proba(X_sub_comp)[:, 1]
    pred_det = model_detailed.predict_proba(X_sub_det)[:, 1]

    # --- THE ENSEMBLE FIX ---
    # Apply the mathematically optimal blend discovered by the CV
    final_pred = (best_weight * pred_comp) + ((1 - best_weight) * pred_det)
    final_pred = np.clip(final_pred, 1e-5, 1 - 1e-5)

    out = pd.DataFrame({"ID": sub["ID"], "Pred": final_pred})
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
