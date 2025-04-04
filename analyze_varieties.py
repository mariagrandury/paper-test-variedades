from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better visualizations
plt.style.use("seaborn-v0_8")  # Using a valid style name
sns.set_palette(["#2b83ba", "#fdae61"])  # Blue and orange palette

# Create output directory for plots
Path("plots").mkdir(exist_ok=True)

# Read the data
df = pd.read_csv("combined_varieties.csv")
cereal_df = pd.read_csv("cereal.csv")


# Get list of varieties from column names
def extract_variety(col_name):
    if "puntuacion_" in col_name:
        # Extract the part between 'puntuacion_' and '_sobre_30'
        return col_name.split("puntuacion_")[1].split("_sobre_30")[0]
    return None


varieties = [v for v in [extract_variety(col) for col in df.columns] if v is not None]


def create_performance_overview():
    """Create overview plots of model performance across varieties."""

    # Calculate mean scores and standard deviation for each model
    mean_scores = pd.DataFrame()
    for metric in ["puntuacion", "aciertos"]:
        cols = [col for col in df.columns if metric in col]
        mean_scores[f"mean_{metric}"] = df[cols].mean(axis=1)
        mean_scores[f"std_{metric}"] = df[cols].std(axis=1)
    mean_scores["modelos"] = df["modelos"]

    # Plot mean scores with error bars
    plt.figure(figsize=(12, 6))
    x = np.arange(len(mean_scores))
    plt.errorbar(
        x,
        mean_scores["mean_puntuacion"],
        yerr=mean_scores["std_puntuacion"],
        fmt="o",
        label="Puntuación",
        color="#2b83ba",  # Blue
    )
    plt.errorbar(
        x,
        mean_scores["mean_aciertos"],
        yerr=mean_scores["std_aciertos"],
        fmt="o",
        label="Aciertos",
        color="#fdae61",  # Orange
    )
    plt.xticks(x, mean_scores["modelos"], rotation=45, ha="right")
    plt.ylabel("Puntuación (sobre 30)")
    plt.title("Puntuación Media en Todas las Variedades")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/lineplot_puntuacion_media_variedades.png")
    plt.close()


def create_heatmap():
    """Create heatmap of model performance across varieties."""

    # Prepare data for heatmap
    score_cols = [col for col in df.columns if "puntuacion" in col]
    score_data = df[["modelos"] + score_cols].set_index("modelos")
    score_data.columns = [
        col.split("puntuacion_")[1].split("_sobre_30")[0] for col in score_data.columns
    ]

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        score_data,
        annot=True,
        cmap="RdYlBu_r",  # Blue to orange colormap
        fmt=".1f",
        center=15,
        vmin=0,
        vmax=30,
    )
    plt.title("Mapa de Calor de Puntuaciones por Variedad")
    plt.xlabel("")  # Remove x-label
    plt.ylabel("")  # Remove y-label
    plt.tight_layout()
    plt.savefig("plots/mapa_calor_puntuaciones.png")
    plt.close()


def analyze_error_patterns():
    """Analyze and visualize error patterns."""

    # Calculate mean error rates
    morfo_cols = [col for col in df.columns if "morfosintaxis" in col]
    lexico_cols = [col for col in df.columns if "lexico" in col]

    mean_errors = pd.DataFrame(
        {
            "modelos": df["modelos"],
            "mean_morfosintaxis": df[morfo_cols].mean(axis=1),
            "mean_lexico": df[lexico_cols].mean(axis=1),
        }
    )

    # Plot error rates
    plt.figure(figsize=(12, 6))
    x = np.arange(len(mean_errors))
    width = 0.35
    plt.bar(
        x - width / 2,
        mean_errors["mean_morfosintaxis"],
        width,
        label="Morfosintaxis",
        color="#2b83ba",  # Blue
    )
    plt.bar(
        x + width / 2,
        mean_errors["mean_lexico"],
        width,
        label="Léxico",
        color="#fdae61",  # Orange
    )
    plt.xticks(x, mean_errors["modelos"], rotation=45, ha="right")
    plt.ylabel("Tasa Media de Error (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/barplot_tasas_error_por_modelo.png")
    plt.close()


def analyze_variety_difficulty():
    """Analyze which varieties are more challenging."""

    # Calculate mean scores and std dev per variety
    variety_means = {}
    variety_stds = {}
    for variety in varieties:
        score_col = f"puntuacion_{variety}_sobre_30"
        variety_means[variety] = df[score_col].mean()
        variety_stds[variety] = df[score_col].std()

    # Plot variety difficulty with error bars
    plt.figure(figsize=(10, 6))
    variety_series = pd.Series(variety_means)
    variety_stds_series = pd.Series(variety_stds)
    sorted_indices = variety_series.sort_values().index

    x = np.arange(len(varieties))
    plt.bar(x, variety_series[sorted_indices], color="#2b83ba")  # Blue
    plt.errorbar(
        x,
        variety_series[sorted_indices],
        yerr=variety_stds_series[sorted_indices],
        fmt="none",
        color="#fdae61",  # Orange
        capsize=5,
    )

    plt.ylabel("Puntuación Media (sobre 30)")
    plt.xticks(x, sorted_indices, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/barplot_puntuacion_media_variedades.png")
    plt.close()


def create_summary_table():
    """Create visual tables for summary statistics."""

    # Calculate overall stats
    score_cols = [col for col in df.columns if "puntuacion" in col]
    mean_scores = df[score_cols].mean(axis=1)

    # Create overall performance table
    overall_stats = pd.DataFrame(
        {
            "Modelo": df["modelos"],
            "Puntuación Media": mean_scores,
            "Desviación Estándar": df[score_cols].std(axis=1),
        }
    ).sort_values("Puntuación Media", ascending=False)

    # Plot overall performance table
    plt.figure(figsize=(12, 6))
    plt.axis("off")
    table = plt.table(
        cellText=overall_stats.round(2).values,
        colLabels=overall_stats.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f2f2f2"] * len(overall_stats.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title("Puntuación Media de los Modelos", pad=20)
    plt.tight_layout()
    plt.savefig(
        "plots/tabla_puntuacion_media_modelos.png", bbox_inches="tight", dpi=300
    )
    plt.close()


def analyze_cereal_correlation():
    """Analyze correlation between model performance and cereal data size."""

    country_map = {
        "antillano": ["cu", "do", "pr"],
        "andino": ["pe", "ec", "bo"],
        "caribeno_continental": ["co", "ve"],
        "chileno": ["cl"],
        "mexicano": ["mx", "gt", "cr", "hn", "ni", "pa", "sv"],
        "peninsular": ["es"],
        "rioplatense": ["ar", "uy", "py"],
    }

    # Prepare cereal data
    cereal_sizes = {}
    for variety in varieties:

        # Map variety to country code
        countries = country_map.get(variety, [])

        # Sum the sizes for the countries in each variety
        total_size = 0
        total_docs = 0
        total_words = 0
        for country in countries:
            country_data = cereal_df[cereal_df["file"] == f"cereal.{country}"]
            if not country_data.empty:
                total_size += country_data["size(M)"].iloc[0]
                total_docs += country_data["documents"].iloc[0]
                total_words += country_data["words"].iloc[0]

        cereal_sizes[variety] = {
            "size_MB": total_size,
            "documents": total_docs,
            "words": total_words,
        }

    # Calculate mean performance for each variety
    variety_performance = {}
    for variety in varieties:
        score_col = f"puntuacion_{variety}_sobre_30"
        variety_performance[variety] = df[score_col].mean()

    # Create correlation plot
    data_dict = {
        "Variety": varieties,
        "Mean Score": [variety_performance[v] for v in varieties],
        "Data Size (MB)": [cereal_sizes[v]["size_MB"] for v in varieties],
        "Documents": [cereal_sizes[v]["documents"] for v in varieties],
        "Words": [cereal_sizes[v]["words"] for v in varieties],
    }
    correlation_data = pd.DataFrame(data_dict)
    correlation_data.set_index("Variety", inplace=True)

    # Calculate correlations
    correlations = correlation_data.corr(method="pearson")["Mean Score"].drop(
        "Mean Score"
    )

    # Create scatter plot for the strongest correlation
    strongest_metric = correlations.abs().idxmax()
    # Translate metric name if it's "Words"
    metric_name = (
        "Número de Palabras en CEREAL"
        if strongest_metric == "Words"
        else strongest_metric
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        correlation_data[strongest_metric],
        correlation_data["Mean Score"],
        color="#2b83ba",  # Blue
    )
    for variety in varieties:
        plt.annotate(
            variety,
            (
                correlation_data.loc[variety, strongest_metric],
                correlation_data.loc[variety, "Mean Score"],
            ),
        )

    # Add correlation coefficient
    corr_value = correlation_data[strongest_metric].corr(correlation_data["Mean Score"])
    plt.text(
        0.05,
        0.95,
        f"Correlación: {corr_value:.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.xlabel(metric_name)
    plt.ylabel("Puntuación Media")
    plt.tight_layout()
    plt.savefig("plots/scatterplot_puntuacion_media_vs_datos_cereal.png")
    plt.close()

    # Create data size comparison table with Spanish column names
    size_comparison = pd.DataFrame(
        {
            "Variedad": varieties,
            "Puntuación Media": [variety_performance[v] for v in varieties],
            "Cantidad de Datos (MB)": [cereal_sizes[v]["size_MB"] for v in varieties],
            "Documentos": [cereal_sizes[v]["documents"] for v in varieties],
            "Palabras (M)": [cereal_sizes[v]["words"] / 1_000_000 for v in varieties],
        }
    ).sort_values("Puntuación Media", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.axis("off")
    table = plt.table(
        cellText=size_comparison.round(2).values,
        colLabels=size_comparison.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f2f2f2"] * len(size_comparison.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title("Puntuación Media por Variedad vs Cantidad de Datos", pad=20)
    plt.tight_layout()
    plt.savefig(
        "plots/tabla_variedades_puntuacion_media_cantidad_datos.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def main():
    create_performance_overview()
    create_heatmap()
    analyze_error_patterns()
    analyze_variety_difficulty()
    create_summary_table()
    analyze_cereal_correlation()

    print(
        "¡Análisis completado! Revisa el directorio 'plots' para ver las visualizaciones."
    )


if __name__ == "__main__":
    main()
