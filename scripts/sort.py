import pandas as pd
keys = [
"057f4daa3089",
"2237da940b5c",
"2db4e83591e3",
"54911337de15",
"6cf238440181",
"a04ee9f627f9",
"cf17eaa23789",
"f30c02cea5a6",
"15ebeda55fc9",
"23bb4ede943d",
"31c8ea661704",
"63ad1dbede39",
"7dafa80b5c3d",
"b0320ed0c3a2",
"d55850928ed4",
"f34705f16985",
"18fbefef2142",
"260d230993af",
"4604087c1df3",
"661811024832",
"b04f88d1f85a",
"dbd3e34a840d",
"f3a61e596340",
"1f79eb96f021",
"2bb7ed78ab9a",
"4d340ec8728f",
"68a6f0c8e359",
"9d2d94a36bc6",
"c69bf557af05",
"de3ae57fe572",
"f411e68095c7",
]

INPUT_CSV = "analysis/benchmarks/20260429_194249/20260429_194249.csv"
OUTPUT_CSV = "analysis/benchmarks/20260429_194249/per_scene.csv"
SUMMARY_CSV = "analysis/benchmarks/20260429_194249/overall_avg.csv"

df = pd.read_csv(INPUT_CSV)

rank = {key: i for i, key in enumerate(keys)}

df = df.sort_values(
    by="scene_id",
    key=lambda col: col.map(rank)
)

METRIC_COLUMNS = [
    "ate_rmse_m",
    "rpe_trans_rmse_m",
    "rpe_rot_rmse_deg",
]


def frame_weighted_average(group: pd.DataFrame, column: str) -> float:
    valid = group[column].notna() & group["frame_count"].notna()
    values = group.loc[valid, column]
    weights = group.loc[valid, "frame_count"]

    if weights.sum() == 0:
        return float("nan")

    return (values * weights).sum() / weights.sum()


def valid_split_count(group: pd.DataFrame, column: str) -> int:
    valid = group[column].notna() & group["frame_count"].notna()
    return group.loc[valid, "split_idx"].nunique()


def split_count_weighted_average(scene_df: pd.DataFrame, value_column: str, weight_column: str) -> float:
    valid = scene_df[value_column].notna() & scene_df[weight_column].gt(0)
    values = scene_df.loc[valid, value_column]
    weights = scene_df.loc[valid, weight_column]

    if weights.sum() == 0:
        return float("nan")

    return (values * weights).sum() / weights.sum()


mono = df[df["mode"] == "mono"].copy()

rows = []
for scene_id, group in mono.groupby("scene_id", sort=False):
    row = {
        "scene_id": scene_id,
        "num_splits": group["split_idx"].nunique(),
    }

    for column in METRIC_COLUMNS:
        row[column] = frame_weighted_average(group, column)
        row[f"{column}_valid_splits"] = valid_split_count(group, column)

    rows.append(row)

out = pd.DataFrame(
    rows,
    columns=[
        "scene_id",
        "num_splits",
        "ate_rmse_m",
        "ate_rmse_m_valid_splits",
        "rpe_trans_rmse_m",
        "rpe_trans_rmse_m_valid_splits",
        "rpe_rot_rmse_deg",
        "rpe_rot_rmse_deg_valid_splits",
    ],
)

out.to_csv(OUTPUT_CSV, index=False, float_format="%.4f", sep="&")

summary = pd.DataFrame(
    [
        {
            "avg_ate": split_count_weighted_average(
                out,
                "ate_rmse_m",
                "ate_rmse_m_valid_splits",
            ),
            "avg_rpe_t": split_count_weighted_average(
                out,
                "rpe_trans_rmse_m",
                "rpe_trans_rmse_m_valid_splits",
            ),
            "avg_rpe_r": split_count_weighted_average(
                out,
                "rpe_rot_rmse_deg",
                "rpe_rot_rmse_deg_valid_splits",
            ),
        }
    ]
)
summary.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")

print(f"Wrote {len(out)} rows to {OUTPUT_CSV}")
print(f"Wrote averages to {SUMMARY_CSV}")
