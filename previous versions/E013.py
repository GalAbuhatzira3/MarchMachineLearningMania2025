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

# Massey:
MASSEY_ORDINALS = "MMasseyOrdinals.csv"

# Elo:
BASE_ELO_SCORE = 1500


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
    # Winner perspective:
    winner = regular[["Season", "WTeamID", "WScore", "LScore"]].copy()
    winner.columns = ["Season", "TeamID", "PF", "PA"]
    winner["G"] = 1
    winner["W"] = 1

    # Loser perspective:
    loser = regular[["Season", "LTeamID", "LScore", "WScore"]].copy()
    loser.columns = ["Season", "TeamID", "PF", "PA"]
    loser["G"] = 1
    loser["W"] = 0

    # Concatenation of winners and losers:
    games = pd.concat([winner, loser], ignore_index=True)

    # Building the feats table
    feats = games.groupby(["Season", "TeamID"], as_index=False).agg(
        G=("G", "sum"),
        W=("W", "sum"),
        PF=("PF", "sum"),
        PA=("PA", "sum"),
    )

    # Convert totals into rates:
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
        # Adjust counting stats to a flat 40-minute baseline
        adj_ot = (40.0 + 5.0 * temp_df["NumOT"]) / 40.0

        cols_to_adjust = [
            "PF", "PA", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PFoul",
            "OppFGM", "OppFGA", "OppFGM3", "OppFGA3", "OppFTM", "OppFTA", "OppOR", "OppDR", "OppTO"
        ]

        for col in cols_to_adjust:
            temp_df[col] = temp_df[col] / adj_ot

        # Now calculate possessions using the NORMALIZED stats
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


def build_elo_end_features(regular: pd.DataFrame, k: float = 20.0, base: float = 1500.0, reversion: float = 0.75,
                           hca: float = 75.0) -> pd.DataFrame:
    games = regular[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"]].copy()
    games = games.sort_values(["Season", "DayNum"], kind="mergesort")

    def expected(a_rating: float, b_rating: float) -> float:
        return 1.0 / (1.0 + 10 ** ((b_rating - a_rating) / 400.0))

    rows = []
    current_season = None
    ratings: dict[int, float] = {}

    for season, day, winner, loser, w_score, l_score, w_loc in games.itertuples(index=False):
        if current_season is None or season != current_season:
            if current_season is not None:
                for tid, r in ratings.items():
                    rows.append((current_season, tid, r))
                for tid in ratings:
                    ratings[tid] = (reversion * ratings[tid]) + ((1.0 - reversion) * base)
            current_season = season

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

        mov = w_score - l_score
        mov_multiplier = np.log(mov) + 1.0
        dynamic_k = k * mov_multiplier

        ratings[winner] = winner_rating + dynamic_k * (1.0 - expected_winner)
        ratings[loser] = loser_rating + dynamic_k * (0.0 - expected_loser)

    if current_season is not None:
        for tid, r in ratings.items():
            rows.append((current_season, tid, r))

    elo = pd.DataFrame(rows, columns=["Season", "TeamID", "EloEnd"])
    elo = elo.sort_values(["Season", "TeamID"]).drop_duplicates(["Season", "TeamID"], keep="last")

    return elo


def get_seeds() -> pd.DataFrame:
    usecols = ["Season", "Seed", "TeamID"]
    men = read_csv(MEN_SEEDS, usecols=usecols)
    women = read_csv(WOMEN_SEEDS, usecols=usecols)
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
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir:      {DATA_DIR}")

    # --------------------------------- Loading Data ----------------------------------------------------------------- #
    regular_compact = get_regular(is_detailed=False)
    regular_detailed = get_regular(is_detailed=True)
    tournament_compact = get_tournament(is_detailed=False)
    tournament_detailed = get_tournament(is_detailed=True)
    sub = read_csv(SAMPLE_SUB, usecols=["ID", "Pred"])

    # --------------------------------- Feature Building ------------------------------------------------------------- #
    elo = build_elo_end_features(regular_compact)
    seeds = get_seeds()

    # Base Compact Matrix
    features_compact = build_team_season_compact_features(regular_compact)
    features_compact = features_compact.merge(elo, on=["Season", "TeamID"], how="left")
    features_compact = features_compact.merge(seeds, on=["Season", "TeamID"], how="left")

    # Fill NAs for base features
    features_compact["SeedNum"] = features_compact["SeedNum"].fillna(17)
    features_compact["SeedLog"] = features_compact["SeedLog"].fillna(np.log(17))

    # Detailed Matrix
    features_detailed = build_team_season_detailed_features(regular_detailed)
    features_detailed = features_detailed.merge(features_compact, on=["Season", "TeamID"], how="inner")

    # --- THE 3-MODEL SPLIT ---
    # Split the detailed features by gender
    features_det_m = features_detailed[features_detailed["TeamID"] < 2000].copy()
    features_det_w = features_detailed[features_detailed["TeamID"] >= 3000].copy()

    # Apply Massey strictly to Men
    massey = get_massey()
    features_det_m = features_det_m.merge(massey, on=["Season", "TeamID"], how="left")

    # Impute missing KenPom ranks for the few Men's teams that might lack them
    features_det_m["RankLog"] = features_det_m["RankLog"].fillna(100 - 4 * np.log(175.0 + 1) - 175.0 / 22)

    # --------------------------------- Build Supervised Datasets ---------------------------------------------------- #
    X_comp, y_comp, groups_comp = build_training_from_tournament(tournament_compact, features_compact)

    tourney_det_m = tournament_detailed[tournament_detailed["WTeamID"] < 2000].copy()
    X_det_m, y_det_m, groups_det_m = build_training_from_tournament(tourney_det_m, features_det_m)

    tourney_det_w = tournament_detailed[tournament_detailed["WTeamID"] >= 3000].copy()
    X_det_w, y_det_w, groups_det_w = build_training_from_tournament(tourney_det_w, features_det_w)

    print(f"\nTraining Rows - Compact: {len(X_comp):,}, Det Men: {len(X_det_m):,}, Det Women: {len(X_det_w):,}")

    # --------------------------------- Building the Models ---------------------------------------------------------- #
    # Strict adherence to Logistic Regression
    model_compact = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    model_det_m = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    model_det_w = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])

    # --------------------------------- 3-Model CV & Grid Search ----------------------------------------------------- #
    print("\n--- Running 3-Model Cross-Validation & Blend Optimization ---")
    folds = 5
    gkf = GroupKFold(n_splits=folds)

    # Out of fold prediction arrays
    oof_comp_m = np.zeros(len(X_det_m))
    oof_det_m = np.zeros(len(X_det_m))

    oof_comp_w = np.zeros(len(X_det_w))
    oof_det_w = np.zeros(len(X_det_w))

    # Men CV Loop
    for i, (tr_idx, va_idx) in enumerate(gkf.split(X_det_m, y_det_m, groups=groups_det_m), 1):
        val_seasons = set(groups_det_m[va_idx])
        tr_mask_comp = ~np.isin(groups_comp, list(val_seasons))

        # Train and Predict Compact (for Men's holdout)
        model_compact.fit(X_comp.iloc[tr_mask_comp], y_comp[tr_mask_comp])
        X_val_comp_format = X_det_m.iloc[va_idx][X_comp.columns]
        oof_comp_m[va_idx] = model_compact.predict_proba(X_val_comp_format)[:, 1]

        # Train and Predict Detailed Men
        model_det_m.fit(X_det_m.iloc[tr_idx], y_det_m[tr_idx])
        oof_det_m[va_idx] = model_det_m.predict_proba(X_det_m.iloc[va_idx])[:, 1]

    # Women CV Loop
    for i, (tr_idx, va_idx) in enumerate(gkf.split(X_det_w, y_det_w, groups=groups_det_w), 1):
        val_seasons = set(groups_det_w[va_idx])
        tr_mask_comp = ~np.isin(groups_comp, list(val_seasons))

        # Train and Predict Compact (for Women's holdout)
        model_compact.fit(X_comp.iloc[tr_mask_comp], y_comp[tr_mask_comp])
        X_val_comp_format = X_det_w.iloc[va_idx][X_comp.columns]
        oof_comp_w[va_idx] = model_compact.predict_proba(X_val_comp_format)[:, 1]

        # Train and Predict Detailed Women
        model_det_w.fit(X_det_w.iloc[tr_idx], y_det_w[tr_idx])
        oof_det_w[va_idx] = model_det_w.predict_proba(X_det_w.iloc[va_idx])[:, 1]

    # Concatenate all cross-validation predictions to find the global optimal blend
    oof_comp_all = np.concatenate([oof_comp_m, oof_comp_w])
    oof_det_all = np.concatenate([oof_det_m, oof_det_w])
    y_det_all = np.concatenate([y_det_m, y_det_w])

    best_weight = 0.5
    best_brier = 1.0

    print("\n--- Ensemble Blend Grid Search ---")
    for w in np.linspace(0.0, 1.0, 21):
        blend_pred = (w * oof_comp_all) + ((1 - w) * oof_det_all)
        b = brier_score_loss(y_det_all, blend_pred)

        if b < best_brier:
            best_brier = b
            best_weight = w

        if w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            print(f"Weight Compact={w:.2f}, Detailed={1 - w:.2f} --> Brier: {b:.5f}")

    print(f"\n>> BEST BLEND: {best_weight:.2f} Compact / {1 - best_weight:.2f} Detailed (Brier: {best_brier:.5f}) <<")

    # --------------------------------- Final Training & Prediction -------------------------------------------------- #
    model_compact.fit(X_comp, y_comp)
    model_det_m.fit(X_det_m, y_det_m)
    model_det_w.fit(X_det_w, y_det_w)

    season, a, b = parse_submission_ids(sub)

    # 1. Base prediction from the Compact model
    X_sub_comp = build_matchup_features(season, a, b, features_compact)
    pred_comp = model_compact.predict_proba(X_sub_comp)[:, 1]

    # 2. Advanced prediction from the Gender-Split Detailed models
    pred_det = np.zeros(len(sub))
    is_men = (a < 2000).to_numpy()
    is_women = (a >= 3000).to_numpy()

    if is_men.any():
        X_sub_det_m = build_matchup_features(season[is_men], a[is_men], b[is_men], features_det_m)
        pred_det[is_men] = model_det_m.predict_proba(X_sub_det_m)[:, 1]

    if is_women.any():
        X_sub_det_w = build_matchup_features(season[is_women], a[is_women], b[is_women], features_det_w)
        pred_det[is_women] = model_det_w.predict_proba(X_sub_det_w)[:, 1]

    # --- THE ENSEMBLE ---
    final_pred = (best_weight * pred_comp) + ((1 - best_weight) * pred_det)
    final_pred = np.clip(final_pred, 1e-5, 1 - 1e-5)

    out = pd.DataFrame({"ID": sub["ID"], "Pred": final_pred})
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()