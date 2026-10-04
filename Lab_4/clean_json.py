import json
import yfinance as yf

# Вхідний JSON
with open("company_symbol_mapping.json", "r") as f:
    company_symbols_map = json.load(f)

start_date = "2003-07-03"
end_date = "2007-05-04"

valid_map = {}

for symbol, name in company_symbols_map.items():
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if not data.empty:
            valid_map[symbol] = name
            print(f"✅ {symbol} залишено")
        else:
            print(f"⚠️ {symbol} пропущено (немає даних у цей період)")
    except Exception as e:
        print(f"❌ {symbol} помилка: {e}")

# Збереження очищеного списку
with open("company_symbol_mapping_clean.json", "w") as f:
    json.dump(valid_map, f, indent=4)

print("\nЗбережено company_symbol_mapping_clean.json")
