import pandas as pd

for fn in [
    'submissions/submission_champion_timestral_refit_blend_w05.csv',
    'submissions/submission_champion_timestral_refit_blend_w10.csv'
]:
    df = pd.read_csv(fn)
    print(f"Checking {fn}:")
    print(f"  Rows: {len(df)}")
    print(f"  Cols: {list(df.columns)}")
    print(f"  Nulls: {df.isna().sum().sum()}")
    print(f"  Min quantity: {df['quantity'].min()}")
    print(f"  Max quantity: {df['quantity'].max()}")
    print(f"  Non-integer quantity: {(df['quantity'] % 1 != 0).sum()}")
    assert len(df) == 79101, "Invalid length"
    assert list(df.columns) == ["week", "warehouse_name", "article", "quantity"], "Invalid columns"
    assert df["quantity"].min() >= 0, "Negative values"
    assert (df["quantity"] % 1 != 0).sum() == 0, "Non-integer values"
    print("  [PASS] Verified 100% compliant!")
