import pandas as pd
from scipy.stats import mannwhitneyu


Mediterranean  = [
'Apulia_Santa_Sabina_Area1.dat',
'Apulia_Santa_Sabina_Area2.dat',
'Apulia_Santa_Sabina_Area3.dat',
'Apulia_Santa_Sabina_Area4.dat',
'Apulia_day1_Area1.dat',
'Apulia_day1_Area2.dat',
'Apulia_day2_Area1.dat',
'Apulia_day3_Area1.dat',
'Crete_Area1.dat',
#'Sicily.dat',
'Turkey_Region1_Area1.dat',
'Turkey_Region1_Area2.dat',
]


Australia = [
'Australia_Portland_01_Area1.dat',
'Australia_Portland_01_Area2.dat',
'Australia_Portland_01_Area3.dat',
'Australia_Portland_01_Area4.dat',
'Australia_Portland_01_Area5.dat',
'Australia_Portland_01_Area6.dat'
#'Australia_Portland_02_Springs.dat',
#'Australia_Portland_03_MIS5.dat',
#'Australia_Portland_04_West.dat',
#'Australia_Portland_05_West.dat',

]

mediterranean_clean = {name[:-4] for name in Mediterranean}
australia_clean = {name[:-4] for name in Australia}

df= pd.read_csv("table.csv", sep=',')
df_med = df[df["name"].isin(mediterranean_clean)]
df_aus = df[df["name"].isin(australia_clean)]

for column in df.columns[1:]:  
    group1 = df_med[column]
    group2 = df_aus[column]

    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"Kolumna: {column}, p = {p_value:.4f}")

    if p_value < 0.05:
        median1 = group1.median()
        median2 = group2.median()
        
        if median1 > median2:
            print(f"Istotna różnica: Med ({median1:.3f}) > Aus ({median2:.3f})")
        else:
            print(f"Istotna różnica: Med ({median1:.3f}) < Aus ({median2:.3f})")

print("---------------")

df= pd.read_csv("table_2.csv", sep=',') #Dawid
df_med = df[df["name"].isin(mediterranean_clean)]
df_aus = df[df["name"].isin(australia_clean)]

for column in df.columns[1:]:  
    group1 = df_med[column]
    group2 = df_aus[column]

    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"Kolumna: {column}, p = {p_value:.4f}")

    if p_value < 0.05:
        median1 = group1.median()
        median2 = group2.median()
        
        if median1 > median2:
            print(f"Istotna różnica: Med ({median1:.3f}) > Aus ({median2:.3f})")
        else:
            print(f"Istotna różnica: Med ({median1:.3f}) < Aus ({median2:.3f})")