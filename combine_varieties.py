import re

import pandas as pd

# List of varieties and their corresponding files
varieties = {
    "antillano": "variedades/antillano.csv",
    "andino": "variedades/andino.csv",
    "caribeno_continental": "variedades/caribeno_continental.csv",
    "chileno": "variedades/chileno.csv",
    "mexicano": "variedades/mexicano.csv",
    "peninsular": "variedades/peninsular.csv",
    "rioplatense": "variedades/rioplatense.csv",
}


def extract_error_rates(error_str):
    # Extract percentages from string like "Morfosintaxis: 30%, Léxico: 30%"
    morfo = re.search(r"Morfosintaxis: (\d+)%", error_str)
    lexico = re.search(r"Léxico: (\d+)%", error_str)
    return int(morfo.group(1)) if morfo else None, (
        int(lexico.group(1)) if lexico else None
    )


def clean_score(score):
    # Remove /30 from scores like "18.2/30"
    return float(score.split("/")[0])


def clean_hits(hits):
    # Remove /30 from hits like "21/30"
    return int(hits.split("/")[0])


# Initialize an empty dictionary to store results
results = {}

# Process each variety
for variety, file_path in varieties.items():
    df = pd.read_csv(file_path)

    # Process each model in the variety
    for _, row in df.iterrows():
        model = row["Modelos"]
        if model not in results:
            results[model] = {"modelos": model}

        # Add score and hits (removing /30)
        results[model][f"puntuacion_{variety}_sobre_30"] = clean_score(
            row["Puntuación"]
        )
        results[model][f"aciertos_{variety}_sobre_30"] = clean_hits(row["Aciertos"])

        # Extract and add error rates (removing %)
        morfo, lexico = extract_error_rates(row["Frecuencia relativa de fallos"])
        results[model][f"fallos_morfosintaxis_{variety}_porcentaje"] = morfo
        results[model][f"fallos_lexico_{variety}_porcentaje"] = lexico

# Convert results to DataFrame
final_df = pd.DataFrame(list(results.values()))

# Reorder columns to group by variety
column_order = ["modelos"]
for variety in varieties.keys():
    column_order.extend(
        [
            f"puntuacion_{variety}_sobre_30",
            f"aciertos_{variety}_sobre_30",
            f"fallos_morfosintaxis_{variety}_porcentaje",
            f"fallos_lexico_{variety}_porcentaje",
        ]
    )

final_df = final_df[column_order]

# Save to CSV
final_df.to_csv("combined_varieties.csv", index=False)
