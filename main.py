import os
import math
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

# === Konfiguration ===
PAST_DAYS = 90
MIN_OVERLAP_DAYS = 60
PVALUE_THRESHOLD = 0.05
TOP_N = 20
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "DEIN_BOT_TOKEN_HIER")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "DEIN_CHAT_ID_HIER")


# === Symbolquellen ===
def fetch_xetra_symbols():
    """Lädt automatisch die Liste der auf Xetra handelbaren Aktien von der Deutschen Börse."""
    url = "https://www.xetra.com/xetra-en/instruments/shares/list-of-tradable-shares"
    try:
        html = requests.get(url, timeout=15).text
    except Exception as e:
        print("Xetra fetch error:", e)
        return []

    syms = []
    for line in html.splitlines():
        if "XETR" in line and "(" in line:
            part = line.strip().split()[0]
            if len(part) <= 10 and part.isalnum():
                syms.append(part)
    syms = list(dict.fromkeys(syms))
    return syms


def load_symbols():
    """Lädt Symbole aus Xetra oder aus symbols.txt."""
    syms = fetch_xetra_symbols()
    if len(syms) > 200:
        print(f"Geladene Xetra-Symbole: {len(syms)}")
        return syms
    try:
        with open("symbols.txt") as f:
            custom = [x.strip() for x in f if x.strip()]
        print(f"Geladene Symbole aus symbols.txt: {len(custom)}")
        return custom
    except FileNotFoundError:
        print("Keine symbols.txt gefunden, verwende kleine Fallback-Liste.")
        return ["ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE", "SAP.DE", "SIE.DE", "VOW3.DE"]


# === Marktdaten ===
def fetch_prices(tickers, period_days):
    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days * 2)
    data = {}
    chunk = 50
    for i in range(0, len(tickers), chunk):
        group = tickers[i:i + chunk]
        try:
            df = yf.download(group, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                             progress=False, threads=True, group_by='ticker', auto_adjust=True)
            if len(group) == 1:
                sym = group[0]
                data[sym] = df['Close'].dropna()
            else:
                for sym in group:
                    try:
                        data[sym] = df[sym]['Close'].dropna()
                    except Exception:
                        pass
        except Exception:
            continue
    return data


# === Statistik ===
def align_series(s1, s2):
    df = pd.concat([s1, s2], axis=1, join='inner').dropna()
    if len(df) < MIN_OVERLAP_DAYS:
        return None, None
    return df.iloc[:, 0], df.iloc[:, 1]


def hedge_ratio_and_spread(a, b):
    X = add_constant(b.values)
    model = OLS(a.values, X).fit()
    beta = model.params[1]
    spread = a - beta * b
    return beta, spread


def compute_zscore(spread):
    mu, sigma = spread.mean(), spread.std()
    if sigma == 0 or math.isnan(sigma):
        return None
    return (spread.iloc[-1] - mu) / sigma


def evaluate_pairs(data):
    results = []
    symbols = list(data.keys())
    for s1, s2 in combinations(symbols, 2):
        a, b = align_series(data[s1], data[s2])
        if a is None:
            continue
        try:
            pvalue = float(coint(a, b)[1])
            if math.isnan(pvalue):
                continue
            beta, spread = hedge_ratio_and_spread(a, b)
            z = compute_zscore(spread)
            if z is None:
                continue
            results.append({
                "sym_a": s1, "sym_b": s2, "pvalue": pvalue,
                "beta": beta, "zscore": z,
                "spread_mean": float(spread.mean()),
                "spread_std": float(spread.std()),
                "n": len(spread)
            })
        except Exception:
            continue
    return pd.DataFrame(results)


# === Reporting ===
def generate_markdown_report(df_final, ts):
    md = f"# Daily Pairs Report {ts}\n\n"
    md += "| Sym A | Sym B | p-value | β | z-Score |\n"
    md += "|-------|-------|---------|----|--------|\n"
    for _, r in df_final.iterrows():
        md += f"| {r.sym_a} | {r.sym_b} | {r.pvalue:.4f} | {r.beta:.4f} | {r.zscore:.2f} |\n"
    path = f"{OUTPUT_DIR}/pairs_{ts}.md"
    with open(path, "w") as f:
        f.write(md)
    return path


def send_telegram_message(text):
    if TELEGRAM_BOT_TOKEN.startswith("DEIN_"):
        print("Telegram nicht konfiguriert.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print("Telegram send failed:", r.text)
    except Exception as e:
        print("Telegram error:", e)


# === Hauptlogik ===
def main():
    universe = load_symbols()
    print(f"Universum: {len(universe)} Symbole")

    data = fetch_prices(universe, PAST_DAYS)
    if len(data) < 5:
        print("Zu wenige Kursdaten.")
        return

    df_pairs = evaluate_pairs(data)
    if df_pairs.empty:
        print("Keine Paare gefunden.")
        return

    df_sig = df_pairs[df_pairs["pvalue"] <= PVALUE_THRESHOLD]
    if df_sig.empty:
        df_sig = df_pairs.sort_values("pvalue").head(TOP_N)

    df_sig["abs_z"] = df_sig["zscore"].abs()
    df_final = df_sig.sort_values("abs_z", ascending=False).head(TOP_N)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    md_path = generate_markdown_report(df_final, ts)
    print(f"Report gespeichert: {md_path}")

    msg_lines = df_final.head(5).to_string(index=False)
    msg = f"*Top Pairs {ts}*\n```\n{msg_lines}\n```\n"
    send_telegram_message(msg)


if __name__ == "__main__":
    main()
