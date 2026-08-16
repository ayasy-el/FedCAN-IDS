from itertools import product
from pathlib import Path

import polars as pl

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# Default: tidak digabung (lihat catatan CAPTURE GROUP di atas).
# Contoh kalau nanti ada alasan valid untuk menggabungkan sesi tertentu:
# SESSION_TO_GROUP = {
#     "Pre_train_D_1": "some_group",
#     "Pre_train_S_1": "some_group",
# }
SESSION_TO_GROUP: dict[str, str] = {}

input_path = Path("data/interim/car_hacking_with_session.parquet")
output_dir = Path("data/processed/car_hacking")
output_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(input_path)

# capture_group = session_id kecuali di-override lewat SESSION_TO_GROUP
sessions_seen = df["session_id"].unique().to_list()
group_map = {s: SESSION_TO_GROUP.get(s, s) for s in sessions_seen}

df = df.with_columns(pl.col("session_id").replace(group_map).alias("capture_group"))

expected_classes = set(df["Class"].unique().to_list())
total_rows = len(df)

group_stats = (
    df.group_by("capture_group")
    .agg(
        pl.len().alias("n_rows"),
        pl.col("Class").unique().alias("classes"),
    )
    .sort("capture_group")
    .to_dicts()
)

n_groups = len(group_stats)
print(f"Jumlah capture_group: {n_groups}")
for g in group_stats:
    print(
        f"  {g['capture_group']:30} n_rows={g['n_rows']:>10,}  classes={sorted(g['classes'])}"
    )

targets = [TRAIN_FRAC, VAL_FRAC, TEST_FRAC]
split_names = ["train", "val", "test"]

if 3**n_groups > 200_000:
    raise RuntimeError(
        f"Jumlah capture_group ({n_groups}) terlalu besar untuk exhaustive "
        f"search (3^{n_groups} kombinasi). Ganti strategi pencarian jadi "
        f"greedy/heuristik, atau gabungkan capture_group yang terlalu kecil."
    )

best_assignment = None
best_score = float("inf")
best_counts = None

for assignment in product(range(3), repeat=n_groups):
    # Pastikan ketiga split terpakai (tidak ada split yang kosong sesi).
    if set(assignment) != {0, 1, 2}:
        continue

    row_counts = [0, 0, 0]
    class_coverage = [set(), set(), set()]

    for split_idx, group in zip(assignment, group_stats):
        row_counts[split_idx] += group["n_rows"]
        class_coverage[split_idx].update(group["classes"])

    # Train, val, dan test WAJIB memiliki seluruh kelas yang ada di
    # dataset -- ini validasi keras, bukan cuma preferensi.
    if any(classes != expected_classes for classes in class_coverage):
        continue

    fractions = [count / total_rows for count in row_counts]

    score = sum((actual - target) ** 2 for actual, target in zip(fractions, targets))

    if score < best_score:
        best_score = score
        best_assignment = assignment
        best_counts = row_counts

if best_assignment is None:
    raise RuntimeError(
        "Tidak ditemukan pembagian capture_group yang membuat semua kelas "
        "tersedia pada train, val, DAN test sekaligus.\n"
        "Opsi:\n"
        "  1. Jalankan audit relative_position (lihat diskusi sebelumnya) "
        "untuk konfirmasi kelas mana yang burst-nya sangat sempit.\n"
        "  2. Pertimbangkan strict chronological-tail split sebagai eval "
        "tambahan (bukan test utama) untuk kelas yang memang tidak bisa "
        "didapat di whole-session holdout manapun.\n"
        "  3. Kurangi jumlah kelas yang divalidasi wajib hadir di test "
        "(mis. kalau satu kelas memang cuma ada di 1 sesi kecil, terima "
        "test tanpa kelas itu dan dokumentasikan limitasinya secara "
        "eksplisit alih-alih memaksakan)."
    )

groups_by_split = {"train": [], "val": [], "test": []}

for split_idx, group in zip(best_assignment, group_stats):
    groups_by_split[split_names[split_idx]].append(group["capture_group"])

print("\nPembagian capture_group:")
for name in split_names:
    print(f"  {name:5}: {groups_by_split[name]}")

print("\nProporsi baris:")
for name, count in zip(split_names, best_counts):
    print(f"  {name:5}: {count:,} ({count / total_rows:.2%})")


def select_groups(group_ids: list[str]) -> pl.DataFrame:
    return df.filter(pl.col("capture_group").is_in(group_ids)).sort(
        ["session_id", "Timestamp"]
    )


train_df = select_groups(groups_by_split["train"])
val_df = select_groups(groups_by_split["val"])
test_df = select_groups(groups_by_split["test"])

splits = {"train": train_df, "val": val_df, "test": test_df}

print("\nDistribusi kelas per split:")
for name, split_df in splits.items():
    actual_classes = set(split_df["Class"].unique().to_list())
    missing_classes = expected_classes - actual_classes

    # Validasi keras: hentikan pipeline kalau ternyata ada kelas hilang
    # (seharusnya sudah dijamin oleh pencarian di atas, ini pengaman ganda).
    if missing_classes:
        raise RuntimeError(
            f"Split '{name}' kehilangan kelas: {sorted(missing_classes)}"
        )

    print(f"\n-- {name} --")
    print(split_df.group_by("Class").len().rename({"len": "count"}).sort("Class"))

train_df.write_parquet(output_dir / "train.parquet", compression="zstd")
val_df.write_parquet(output_dir / "val.parquet", compression="zstd")
test_df.write_parquet(output_dir / "test.parquet", compression="zstd")

print(
    "\nOK: semua kelas hadir di train, val, dan test. Tidak ada session_id yang terpecah antar split."
)
print(
    "\nCATATAN: karena val/test kemungkinan cuma berisi 1-2 capture_group, "
    "pertimbangkan menjalankan Leave-One-Group-Out (ulangi training/eval "
    "dengan beberapa kombinasi valid berbeda, bandingkan mean +/- std "
    "metrik) sebelum menyimpulkan performa model dari kombinasi ini saja."
)
