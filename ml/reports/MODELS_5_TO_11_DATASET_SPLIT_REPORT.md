# Models 5-11 Dataset Split Report

**Dataset Source**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\dataset\plantvillage dataset\color`
**Random Seed**: 42
**Splitting Methodology**: Stratified grouping by exact image hash.
**Duplicate Grouping Methodology**: SHA-256 content hashing to ensure exact duplicates always fall in the same split (Train/Val/Test), preventing data leakage.

## Model 5 — APPLE
- **Classes**: 4
- **Class Mapping**: {'0': 'healthy', '1': 'apple_scab', '2': 'black_rot', '3': 'cedar_apple_rust'}
- **Total Images**: 3171
- **Train**: 2536
- **Validation**: 316
- **Test**: 319
- **Duplicate Groups**: 7
- **Cross-split Leakage**: 0
- **Master CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model5_apple_master.csv`
- **Split CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model5_apple_split.csv`

## Model 6 — CHERRY
- **Classes**: 2
- **Class Mapping**: {'0': 'healthy', '1': 'powdery_mildew'}
- **Total Images**: 1906
- **Train**: 1524
- **Validation**: 190
- **Test**: 192
- **Duplicate Groups**: 0
- **Cross-split Leakage**: 0
- **Master CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model6_cherry_master.csv`
- **Split CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model6_cherry_split.csv`

## Model 7 — CORN
- **Classes**: 4
- **Class Mapping**: {'0': 'healthy', '1': 'cercospora_leaf_spot gray_leaf_spot', '2': 'common_rust_', '3': 'northern_leaf_blight'}
- **Total Images**: 3852
- **Train**: 3080
- **Validation**: 384
- **Test**: 388
- **Duplicate Groups**: 0
- **Cross-split Leakage**: 0
- **Master CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model7_corn_master.csv`
- **Split CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model7_corn_split.csv`

## Model 8 — GRAPE
- **Classes**: 4
- **Class Mapping**: {'0': 'healthy', '1': 'black_rot', '2': 'esca_(black_measles)', '3': 'leaf_blight_(isariopsis_leaf_spot)'}
- **Total Images**: 4062
- **Train**: 3248
- **Validation**: 405
- **Test**: 409
- **Duplicate Groups**: 0
- **Cross-split Leakage**: 0
- **Master CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model8_grape_master.csv`
- **Split CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model8_grape_split.csv`

## Model 9 — PEACH
- **Classes**: 2
- **Class Mapping**: {'0': 'healthy', '1': 'bacterial_spot'}
- **Total Images**: 2657
- **Train**: 2125
- **Validation**: 265
- **Test**: 267
- **Duplicate Groups**: 0
- **Cross-split Leakage**: 0
- **Master CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model9_peach_master.csv`
- **Split CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model9_peach_split.csv`

## Model 10 — PEPPER
- **Classes**: 2
- **Class Mapping**: {'0': 'healthy', '1': 'bacterial_spot'}
- **Total Images**: 2475
- **Train**: 1979
- **Validation**: 246
- **Test**: 250
- **Duplicate Groups**: 0
- **Cross-split Leakage**: 0
- **Master CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model10_pepper_master.csv`
- **Split CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model10_pepper_split.csv`

## Model 11 — STRAWBERRY
- **Classes**: 2
- **Class Mapping**: {'0': 'healthy', '1': 'leaf_scorch'}
- **Total Images**: 1565
- **Train**: 1251
- **Validation**: 155
- **Test**: 159
- **Duplicate Groups**: 0
- **Cross-split Leakage**: 0
- **Master CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model11_strawberry_master.csv`
- **Split CSV**: `C:\Users\ak021\Desktop\workspace\plantDx\plant image detector\metadata\model11_strawberry_split.csv`

## Global Verification
- **Missing files**: 0
- **Unreadable images**: 0
- **Cross-class duplicates**: 0
- **Cross-split duplicate leakage**: 0
- **Classes missing from train**: 0
- **Classes missing from validation**: 0
- **Classes missing from test**: 0

## Test Set Status
**LOCKED** - Test sets must not be used for anything other than final evaluation.