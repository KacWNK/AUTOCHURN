import logging
import matplotlib.pyplot as plt
from fpdf import FPDF
from FeatureAnalyzer import FeatureAnalyzer
from ModelOptimizer import ModelOptimizer
from ModelValidator import ModelValidator
from sklearn.metrics import classification_report
import seaborn as sns

def generate_summary_report(df, target_column, output_dir="./figures"):
    """
    Funkcja generuje raport podsumowujący cały proces analizy, optymalizacji i walidacji modelu.

    Parametry:
        df (pd.DataFrame): Dane wejściowe.
        target_column (str): Nazwa kolumny docelowej.
        output_dir (str): Katalog, w którym zapisywane są wykresy i raporty.

    Zwraca:
        None
    """
    logging.info("Rozpoczęcie generowania raportu.")

    # 1. Eksploracyjna Analiza Danych (EDA)
    logging.info("Eksploracyjna analiza danych (EDA).")
    eda_summary = df.describe(include='all')
    logging.info(f"Podstawowe statystyki danych:\n{eda_summary}")
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Histogramy dla cech numerycznych
    hist_paths = []
    for col in numeric_features:
        hist_path = f"{output_dir}/hist_{col}.png"
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='blue')
        plt.title(f"Rozkład cechy: {col}")
        plt.savefig(hist_path, format='png')
        plt.close()
        hist_paths.append(hist_path)

    # Wykresy słupkowe dla cech kategorycznych
    bar_paths = []
    for col in categorical_features:
        bar_path = f"{output_dir}/bar_{col}.png"
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar', color='orange')
        plt.title(f"Rozkład cechy kategorycznej: {col}")
        plt.ylabel("Liczność")
        plt.savefig(bar_path, format='png')
        plt.close()
        bar_paths.append(bar_path)

    # 2. Analiza cech
    logging.info("Rozpoczęcie analizy cech.")
    analyzer = FeatureAnalyzer(df, target_column)

    logging.info("Korelacja cech z celem.")
    high_corr_features = analyzer.correlation_with_target()

    logging.info("Generowanie macierzy korelacji.")
    removed_corr_features = analyzer.correlation_matrix()
    logging.info(f"Usunięte cechy o wysokiej korelacji między sobą: {removed_corr_features}")

    logging.info("Obliczanie ważności cech.")
    feature_importances = analyzer.feature_importance()
    logging.info(f"Ważności cech:\n{feature_importances}")


    logging.info("Tworzenie golden features.")
    golden_features_df = analyzer.generate_golden_features(feature_importances.index[0])
    logging.info(f"Utworzone golden features:\n{golden_features_df.columns.tolist()}")

    # 3. Porównanie modeli za pomocą get_models_candidates
    logging.info("Porównanie modeli za pomocą funkcji get_models_candidates.")
    optimizer = ModelOptimizer(df, target_column)
    optimizer.get_models_candidates()
    logging.info("Zapisano wykres porównania modeli jako ./figures/boxplot.png.")

    # Dynamiczne określenie najlepszego modelu na podstawie wyników walidacji
    logging.info("Wybieranie najlepszego modelu na podstawie wyników walidacji.")
    model_scores = {}
    for model in optimizer.get_models().keys():
        scores = optimizer.evaluate_model(optimizer.x_test, optimizer.y_test, optimizer.get_models()[model])
        model_scores[model] = scores.mean()

    best_model = max(model_scores, key=model_scores.get)
    logging.info(f"Najlepszy model: {best_model} z wynikiem {model_scores[best_model]:.4f}.")

    # 4. Walidacja najlepszego modelu
    logging.info("Walidacja najlepszego modelu.")
    best_optimized_model = optimizer.optimize_model(best_model)
    validator = ModelValidator(df, target_column, best_optimized_model)
    validator.fit_model()

    logging.info("Generowanie krzywej ROC.")
    auc, gini = validator.get_roc_plot()
    logging.info(f"Zapisano wykres krzywej ROC jako {output_dir}/roc_curve.png.")

    logging.info("Generowanie raportu klasyfikacji.")
    y_pred = validator.model.predict(validator.x_test)
    clf_report = classification_report(validator.y_test, y_pred, output_dict=True)

    # Przygotowanie tabeli z raportem klasyfikacyjnym
    clf_table_rows = [["", "Precision", "Recall", "F1-Score", "Support"]]
    for label, metrics in clf_report.items():
        if isinstance(metrics, dict):
            row = [
                f"{label}",
                f"{metrics['precision']:.2f}",
                f"{metrics['recall']:.2f}",
                f"{metrics['f1-score']:.2f}",
                f"{int(metrics['support'])}"
            ]
            clf_table_rows.append(row)

    logging.info(f"Raport klasyfikacyjny:\n{clf_report}")
    
    # Słownik z pełnymi nazwami modeli
    model_names = {
        "lr": "Regresja Logistyczna",
        "rf": "Las Losowy",
        "cart": "Drzewo Decyzyjne",
        "knn": "K-Najblizszych Sasiadow",
        "xgboost": "XGBoost",
        
    }
    best_model_name = model_names[best_model]


    # 5. Tworzenie PDF z raportem
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(0, 10, "PODSUMOWANIE RAPORTU", ln=True, align='C')
    
    # Obliczenie szerokości tabeli
    first_table_col_widths = [70, 70]
    total_first_table_width = sum(first_table_col_widths)
    start_x_first_table = (pdf.w - total_first_table_width) / 2  # Wyśrodkowanie

    # Kolor wypelnienia komórek
    pdf.set_fill_color(200, 220, 255)
    # Ustawienie początkowej pozycji X
    pdf.set_x(start_x_first_table)
    # Dodawanie komórek do pierwszej tabeli
    pdf.cell(first_table_col_widths[0], 10, "Najlepszy model", border=1, fill=True, align='C')
    pdf.cell(first_table_col_widths[1], 10, f"{best_model_name}", border=1, align='C')
    pdf.ln(10)
    pdf.set_x(start_x_first_table)
    pdf.cell(first_table_col_widths[0], 10, "Wynik walidacji", border=1, fill=True, align='C')
    pdf.cell(first_table_col_widths[1], 10, f"{model_scores[best_model]:.4f}", border=1, align='C')
    pdf.ln(10)
    pdf.set_x(start_x_first_table)
    pdf.cell(first_table_col_widths[0], 10, "AUROC", border=1, fill=True, align='C')
    pdf.cell(first_table_col_widths[1], 10, f"{auc:.4f}", border=1, align='C')
    pdf.ln(10)
    pdf.set_x(start_x_first_table)
    pdf.cell(first_table_col_widths[0], 10, "Gini", border=1, fill=True, align='C')
    pdf.cell(first_table_col_widths[1], 10, f"{gini:.4f}", border=1, align='C')
    pdf.ln(20)
    
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Raport klasyfikacji:", ln=True, align='C')
    pdf.set_font("Arial", size=10)

    line_height = pdf.font_size * 1.5
    col_widths = [40, 30, 30, 30, 30]
    total_table_width = sum(col_widths)
    start_x_second_table = (pdf.w - total_table_width) / 2  # Wyśrodkowanie

# Dodawanie wierszy do drugiej tabeli
    for row in clf_table_rows:
        pdf.set_x(start_x_second_table)
        for idx, cell in enumerate(row):
            pdf.cell(col_widths[idx], line_height, cell, border=1, align='C')
        pdf.ln(line_height)
    ##############################################################################################################
    pdf.ln(20)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Statystyki opisowe:", ln=True, align='C')
    pdf.set_font("Arial", size=10)

    desc = eda_summary.round(2).transpose()
    line_height = pdf.font_size * 1.5
    col_widths = [40] + [20] * (len(desc.columns))  # Szerokość kolumn

    # Dodanie nagłówków
    desc=desc.drop(columns='count')
    pdf.set_font("Arial", style='B', size=10)
    headers = ["Feature"] + list(desc.columns)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], line_height, header, border=1, align='C')
    pdf.ln(line_height)

    # Dodanie wierszy z wartościami
    pdf.set_font("Arial", size=10)
    for feature, values in desc.iterrows():
        pdf.cell(col_widths[0], line_height, feature, border=1, align='C')  # Nazwa cechy
        for i, value in enumerate(values):
            pdf.cell(col_widths[i + 1], line_height, str(value), border=1, align='C')
        pdf.ln(line_height)
    ##############################################################################################################
    # EDA
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Eksploracyjna analiza danych:", ln=True)

    images = hist_paths + bar_paths
    images_per_row = 3
    x_offset = 10
    y_offset = 30
    img_width = 60
    img_height = 40

    for i, img_path in enumerate(images):
        if i % images_per_row == 0 and i != 0:
            y_offset += img_height + 10
            x_offset = 10
        pdf.image(img_path, x=x_offset, y=y_offset, w=img_width, h=img_height)
        x_offset += img_width + 10
        if y_offset + img_height + 10 > 270:  # Nowa strona, jeśli brak miejsca
            pdf.add_page()
            y_offset = 30
            x_offset = 10


    ##############################################################################################################
    # Wykresy z analizy
    pdf.add_page()
    pdf.cell(0, 10, "Korelacja cech z celem:", ln=True)
    pdf.image("./figures/correlation_with_target.png", x=10, y=20, w=180)
    pdf.ln(150)
    pdf.multi_cell(0, 10, f"Wnioski: Wykres pokazuje, ktore cechy sa skorelowane z celem. Wartosci bliskie 1 oznaczaja silna korelacje dodatnia, a bliskie -1 silna korelacje ujemna. Cechy o wysokiej korelacji z celem moga byc kluczowe w modelowaniu. Wartosci ponizej 0.1 sa zazwyczaj uznawane za niskie i nieistotne. Najwyzsza korelacje z celem mialy cechy: {high_corr_features}.")

    pdf.add_page()
    pdf.cell(0, 10, "Macierz korelacji cech:", ln=True)
    pdf.image("./figures/correlation_matrix.png", x=10, y=20, w=180)
    pdf.ln(150)
    pdf.multi_cell(0, 10, f"Wnioski: Wykres pokazuje, ktore cechy sa ze soba skorelowane. W wyniku analizy usunieto cechy o wysokiej korelacji miedzy soba: {removed_corr_features}.")

    pdf.add_page()
    pdf.cell(0, 10, "Waznosc cech:", ln=True)
    pdf.image("./figures/feature_importance.png", x=10, y=20, w=180)
    pdf.ln(150)
    pdf.multi_cell(0, 10, "Wnioski: Wykres pokazuje, ktore cechy mialy najwiekszy wplyw na predykcje modelu. Mozna je rozwazyc jako kluczowe w dalszych analizach.")

    pdf.add_page()
    pdf.cell(0, 10, "Porownanie modeli:", ln=True)
    pdf.image('./figures/boxplot.png', x=10, y=20, w=180)
    pdf.ln(150)
    pdf.multi_cell(0, 10, f"Wnioski: Wykres przedstawia porownanie roznych modeli pod wzgledem ich skutecznosci. Widac, ze model '{best_model}' osiagnal najwyzsze wyniki w walidacji. Z wynikiem {model_scores[best_model]:.4f} jest to najlepszy model do dalszych analiz.")

    pdf.add_page()
    pdf.cell(0, 10, "Krzywa ROC:", ln=True)
    pdf.image("./figures/roc_curve.png", x=10, y=20, w=180)

    pdf.output(f"{output_dir}/report_summary.pdf")

    logging.info(f"Raport został zapisany jako {output_dir}/report_summary.pdf.")
