import pandas as pd

def xlsx_to_csv(xlsx_path, csv_path, sheet_name=0):
    # Lee el archivo xlsx
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    # Guarda como csv
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # Ejemplo de uso:
    xlsx_file = ".\\nn\\predecir_readmision_uci.xlsx"  # Cambia esto por el nombre de tu archivo xlsx
    csv_file = "predecir_readmision_uci.csv"   # Cambia esto por el nombre de salida deseado
    xlsx_to_csv(xlsx_file, csv_file)
