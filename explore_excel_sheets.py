import pandas as pd

# Load the Excel file and check all sheets
excel_file = 'SubscriptionUseCase_Dataset.xlsx'

print("Exploring all sheets in the Excel file...")

# Get all sheet names
xl_file = pd.ExcelFile(excel_file)
sheet_names = xl_file.sheet_names

print(f"\nFound {len(sheet_names)} sheets:")
for i, sheet in enumerate(sheet_names):
    print(f"{i+1}. {sheet}")

print("\n" + "="*50)

# Load and examine each sheet
all_data = {}
for sheet in sheet_names:
    print(f"\n=== SHEET: {sheet} ===")
    df = pd.read_excel(excel_file, sheet_name=sheet)
    all_data[sheet] = df
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print("\n" + "-"*30)

# Save each sheet as separate CSV for easier access
for sheet_name, df in all_data.items():
    filename = f"{sheet_name.replace(' ', '_').lower()}_data.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {sheet_name} as {filename}")