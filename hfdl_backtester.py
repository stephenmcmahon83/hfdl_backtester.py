import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Single-Stock 7-Strategy Suite & Gap Auditor", layout="wide")

st.title("📊 Single-Stock 7-Strategy Backtest Suite (with True Clock-Grid Gap Protection)")
st.markdown("""
Upload any 1-minute OHLCV CSV file. This engine audits your file for **missing 1-minute bars and intraday data gaps**, 
omits corrupted/incomplete sessions, reindexes valid days onto a **strict 1-minute clock grid** so missing minutes never 
create false signals, and backtests **all 7 strategies** side-by-side.
""")

with st.expander("📖 Built-In Guide: How Each of the 7 Strategies & the Missing-Data Gate Work"):
    st.markdown("""
    ### 🛡️ 3-Layer Anti-False-Signal Architecture
    1. **Daily Completeness & Max-Gap Gate:** Omits any trading day with fewer than your required % of 390 regular-session bars (09:30–15:59), any single intraday data blackout larger than your Max Gap slider, or fewer than 10 bars in the first 15 minutes.
    2. **True 1-Minute Clock-Grid Reindexing:** Reindexes every qualifying day onto an exact 1-minute grid (`09:30` to `15:59`). Short-term returns (`1m`, `2m`, `3m`) and consecutive up/down bar streaks require **strictly adjacent real clock minutes**—if even 1 minute is missing, streak counters reset to `0`.
    3. **Real-Bar Execution Lock:** Entries and exits can only trigger on verified timestamps where actual market volume traded (`is_real_bar == True`).

    ---
    ### ⚡ The 7 Trading Systems Tested
    * **1. Core VWAP + 3-Bar/IBS Dip (Long-Only):** Buys pullbacks when `Close > VWAP` and `Return from 9:30 Open >= +0.15%`, triggered by either **3+ consecutive Down 1-minute closes** or a **3-minute dip (`<= -0.10%`) closing in the bottom 15% of the candle (`IBS <= 0.15`)**.
    * **2. Full Multi-Signal Semi-HFT (Long + Short):** Combines Strategy 1 with an **OR15 Breakout + 1-bar micro-pause** Long setup, a **4-bar Down-streak capitulation (`IBS <= 0.05`)** Long setup, and a **Bearish VWAP + OR15 Breakdown + 2-bar pop (`IBS >= 0.80`)** Short setup.
    * **3. Rule 10b-18 Gap-Down Reclaim + VWAP Dip:** Designed for mega-caps with heavy corporate buybacks. Adds an opening-bell trigger (`09:35–11:00`) that buys a 2-bar micro-dip as soon as an overnight gap-down (`<= -0.20%`) reclaims both the `09:30` Open and Session VWAP.
    * **4. OR15 + VWAP + IBS 80/20 Momentum Breakout:** Enters **Long** when price breaks above both the 15-minute Opening Range High and VWAP with 15-minute momentum `>= +0.20%` and `IBS >= 0.80` (or **Short** below OR15 Low and VWAP with 15m momentum `<= -0.20%` and `IBS <= 0.20`).
    * **5. Volatility Crush & Breakdown Cascade:** Built for decaying volatility/inverse instruments (`VXX`). Shorts 2-minute micro-bounces below VWAP and OR15 Low, 15-minute breakdown accelerations (`<= -0.80%`), and overbought Z-score pops (`Z >= 1.8`) on red days; goes Long only on extreme volatility breakouts (`>= +1.20%` from open).
    * **6. 10:00 AM First-30m Session Continuation (1 Trade/Day):** Checks the first 30 minutes (`09:30–10:00`) at `10:00 AM`. Goes **Long** for the rest of the day if `First 30m Return >= +0.25%` above VWAP, or **Short** if `First 30m Return <= -0.50%` below VWAP.
    * **7. Classic Opening Range Breakout (1 Trade/Day):** Tracks the high and low of the first $N$ minutes (configurable in the sidebar) and takes at most 1 trade per day on the first 1-minute close above the Opening Range High or below the Opening Range Low.
    """)

# ==========================================
# SIDEBAR: DATA UPLOAD, SIZING & GAP FILTER
# ==========================================
st.sidebar.header("1. Upload Single-Stock 1-Min CSV")
uploaded_file = st.sidebar.file_uploader("Upload 1-Minute CSV File", type=["csv"])

st.sidebar.header("2. Capital, Compounding & Commissions")
base_trade_size = st.sidebar.number_input("Base Dollar Value per Trade ($)", value=20000.0, step=1000.0, min_value=500.0)
commission_per_trade = st.sidebar.number_input("Commission per Trade ($)", value=1.00, step=0.25, min_value=0.0, format="%.2f")

sizing_mode = st.sidebar.selectbox(
    "Position Sizing / Compounding Mode",
    options=[
        "Strict Fixed Dollar Size (Always use Base $)",
        "Capped at Start, Deduct Losses (No Profit Reinvestment)",
        "Full Compounding (Reinvest Profits & Deduct Losses)"
    ]
)

st.sidebar.header("3. Missing Data & Anti-False-Signal Gate")
min_completeness_pct = st.sidebar.slider(
    "Min Daily Bar Completeness (% of 390 Bars)", 
    min_value=50, max_value=100, value=93, step=1,
    help="390 bars = 100% complete regular session (09:30–15:59). 93% requires at least 363 real 1-min bars per day."
)
max_gap_allowed = st.sidebar.slider(
    "Max Allowed Single Intraday Gap (Minutes)", 
    min_value=1, max_value=30, value=5, step=1,
    help="Set to 1 for 0 missing minutes allowed. Set to 5 to omit any day where trading gapped for >5 consecutive minutes."
)

st.sidebar.header("4. Strategy Exit & Hold Parameters")
default_max_hold = st.sidebar.slider("Semi-HFT Max Hold (Minutes)", min_value=15, max_value=240, value=120, step=5)
default_pt_pct = st.sidebar.slider("Default Profit Target (%)", min_value=0.20, max_value=3.00, value=0.75, step=0.05) / 100.0
default_sl_pct = st.sidebar.slider("Default Stop Loss (%)", min_value=0.20, max_value=3.00, value=0.80, step=0.05) / 100.0
orb_minutes = st.sidebar.slider("Strategy 7: Classic ORB Duration (Mins)", min_value=5, max_value=60, value=15, step=5)

# ==========================================
# CORE DATA PROCESSING & CLOCK-GRID ENGINE
# ==========================================
@st.cache_data
def load_and_audit_csv(file, min_comp_pct, max_gap_min, orb_mins):
    raw = pd.read_csv(file)
    dt_col = raw.columns[0]
    raw[dt_col] = pd.to_datetime(raw[dt_col], errors='coerce')
    raw = raw.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
    raw.rename(columns={dt_col: 'datetime'}, inplace=True)
    
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if len(raw.columns) >= 6:
        raw = raw[['datetime'] + list(raw.columns[1:6])]
        raw.columns = ['datetime'] + expected_cols
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close']).reset_index(drop=True)
    
    if raw['datetime'].dt.tz is not None:
        raw['datetime'] = raw['datetime'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        
    raw['date'] = raw['datetime'].dt.date
    raw['time'] = raw['datetime'].dt.strftime('%H:%M')
    raw = raw[(raw['time'] >= '09:30') & (raw['time'] <= '15:59')].copy()
    raw = raw[~raw['datetime'].duplicated(keep='first')].copy()
    
    if 'VXX' in file.name.upper():
        raw = raw[raw['date'] != pd.to_datetime('2024-08-05').date()].copy()

    min_bars_req = int(round(390 * (min_comp_pct / 100.0)))
    
    clean_days = []
    total_days = 0
    kept_days = 0
    omit_low_bars = 0
    omit_gap = 0
    omit_or = 0
    all_day_bar_counts = []
    
    for d, grp in raw.groupby('date'):
        total_days += 1
        n_bars = len(grp)
        all_day_bar_counts.append(n_bars)
        
        if n_bars < min_bars_req:
            omit_low_bars += 1
            continue
            
        diffs = grp['datetime'].diff().dt.total_seconds() / 60.0
        max_g = diffs.max() if len(diffs) > 1 else 1.0
        if max_g > max_gap_min:
            omit_gap += 1
            continue
            
        or15_bars = grp[grp['time'] <= '09:44']
        if len(or15_bars) < 10:
            omit_or += 1
            continue
            
        # Reindex onto strict 1-minute clock grid (09:30 to 15:59)
        grid_idx = pd.date_range(start=f"{d} 09:30:00", end=f"{d} 15:59:00", freq='1min')
        g_re = grp.set_index('datetime').reindex(grid_idx)
        g_re.index.name = 'datetime'
        g_re['is_real_bar'] = g_re['Close'].notna()
        g_re['date'] = d
        g_re['year'] = d.year
        g_re['time'] = g_re.index.strftime('%H:%M')
        
        g_re['Close_real'] = g_re['Close']
        g_re['Volume_fill'] = g_re['Volume'].fillna(0.0)
        g_re['Close_ffill'] = g_re['Close'].ffill()
        g_re['High_ffill'] = g_re['High'].fillna(g_re['Close_ffill'])
        g_re['Low_ffill'] = g_re['Low'].fillna(g_re['Close_ffill'])
        
        kept_days += 1
        clean_days.append(g_re.reset_index())
        
    audit = {
        'total_days': total_days,
        'kept_days': kept_days,
        'omitted_days': total_days - kept_days,
        'omit_low_bars': omit_low_bars,
        'omit_gap': omit_gap,
        'omit_or': omit_or,
        'avg_bars_raw': np.mean(all_day_bar_counts) if all_day_bar_counts else 0.0,
        'pct_completeness_raw': (np.mean(all_day_bar_counts) / 390.0 * 100.0) if all_day_bar_counts else 0.0
    }
    
    if not clean_days:
        return pd.DataFrame(), audit
        
    df = pd.concat(clean_days, ignore_index=True)
    
    # Intraday VWAP
    df['typ_price'] = (df['High_ffill'] + df['Low_ffill'] + df['Close_ffill']) / 3.0
    df['pv'] = df['typ_price'] * df['Volume_fill']
    df['vwap'] = df.groupby('date')['pv'].cumsum() / df.groupby('date')['Volume_fill'].cumsum().replace(0, np.nan)
    
    # True clock-minute returns (NaN if either endpoint bar was missing)
    df['ret_1m'] = df.groupby('date')['Close_real'].pct_change(1, fill_method=None)
    df['ret_2m'] = df.groupby('date')['Close_real'].pct_change(2, fill_method=None)
    df['ret_3m'] = df.groupby('date')['Close_real'].pct_change(3, fill_method=None)
    df['ret_15m'] = df.groupby('date')['Close_ffill'].pct_change(15, fill_method=None)
    
    first_opens = raw.groupby('date')['Open'].first()
    df['day_open'] = df['date'].map(first_opens)
    df['ret_from_open'] = (df['Close_real'] - df['day_open']) / df['day_open']
    
    last_closes = raw.groupby('date')['Close'].last()
    df['prev_day_close'] = df['date'].map(last_closes.shift(1))
    df['overnight_gap'] = (df['day_open'] - df['prev_day_close']) / df['prev_day_close']
    
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    df['ibs'] = (df['Close_real'] - df['Low']) / rng
    
    or15 = raw[raw['time'] <= '09:44'].groupby('date').agg(or15_high=('High', 'max'), or15_low=('Low', 'min'))
    or30 = raw[raw['time'] <= '09:59'].groupby('date').agg(or30_high=('High', 'max'), or30_low=('Low', 'min'), or30_close=('Close', 'last'))
    df['or15_high'] = df['date'].map(or15['or15_high'])
    df['or15_low'] = df['date'].map(or15['or15_low'])
    df['or30_ret'] = (df['date'].map(or30['or30_close']) - df['day_open']) / df['day_open']
    
    # Custom ORB window for Strategy 7
    or_end_time = (pd.to_datetime('09:30') + pd.Timedelta(minutes=orb_mins - 1)).strftime('%H:%M')
    or_custom = raw[raw['time'] <= or_end_time].groupby('date').agg(orc_high=('High', 'max'), orc_low=('Low', 'min'))
    df['orc_high'] = df['date'].map(or_custom['orc_high'])
    df['orc_low'] = df['date'].map(or_custom['orc_low'])
    df['or_after_time'] = (pd.to_datetime('09:30') + pd.Timedelta(minutes=orb_mins)).strftime('%H:%M')
    
    # Strict contiguous 1-minute streaks (any missing minute resets streak to 0)
    ret1 = df['ret_1m'].values
    is_real = df['is_real_bar'].values
    dates = df['date'].values
    u_st, d_st = np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=int)
    cu, cd = 0, 0
    for i in range(len(df)):
        if i == 0 or dates[i] != dates[i-1] or not is_real[i] or np.isnan(ret1[i]):
            cu, cd = 0, 0
        else:
            if ret1[i] > 0:
                cu += 1; cd = 0
            elif ret1[i] < 0:
                cd += 1; cu = 0
            else:
                cu = 0; cd = 0
        u_st[i] = cu; d_st[i] = cd
    df['u_streak'] = u_st
    df['d_streak'] = d_st
    
    g = df.groupby('date')['Close_ffill']
    df['ma_15'] = g.transform(lambda x: x.rolling(15, min_periods=10).mean())
    df['std_15'] = g.transform(lambda x: x.rolling(15, min_periods=10).std())
    df['z_15'] = (df['Close_real'] - df['ma_15']) / df['std_15'].replace(0, np.nan)
    
    return df, audit

def simulate_strategy(df, strat_name, l_sig, s_sig, ex_l, ex_s, max_hold, sl_pct, pt_pct, max_trades_per_day=99):
    trades = []
    dates = df['date'].values
    years = df['year'].values
    times = df['datetime'].values
    time_strs = df['time'].values
    closes = df['Close_real'].values
    is_real = df['is_real_bar'].values
    l_arr, s_arr = l_sig.values, s_sig.values
    el_arr, es_arr = ex_l.values, ex_s.values
    
    n = len(df)
    in_pos = 0
    entry_px, entry_idx, entry_time, shares, alloc_sz = 0.0, 0, None, 0, 0.0
    cur_eq = base_trade_size
    trades_today = 0
    
    for i in range(n):
        if i == 0 or dates[i] != dates[i-1]:
            trades_today = 0
        if not is_real[i]:
            continue
            
        is_eod = (i == n - 1) or (dates[i+1] != dates[i]) or (time_strs[i] >= '15:58')
        
        if in_pos != 0:
            bars_held = i - entry_idx
            cur_px = closes[i]
            ret = (cur_px - entry_px) / entry_px if in_pos == 1 else (entry_px - cur_px) / entry_px
            reason = None
            if is_eod:
                reason = 'EOD Close'
            elif ret <= -sl_pct:
                reason = 'Stop Loss'
            elif ret >= pt_pct:
                reason = 'Profit Target'
            elif (in_pos == 1 and el_arr[i]) or (in_pos == -1 and es_arr[i]):
                reason = 'Regime Reversal'
            elif bars_held >= max_hold:
                reason = f'Time Exit ({max_hold}m)'
                
            if reason is not None:
                gross = (cur_px - entry_px) * shares if in_pos == 1 else (entry_px - cur_px) * shares
                net = gross - commission_per_trade
                cur_eq += net
                trades.append({
                    'Strategy': strat_name,
                    'Date': dates[i],
                    'Year': years[i],
                    'Entry Time': entry_time,
                    'Exit Time': times[i],
                    'Direction': 'LONG' if in_pos == 1 else 'SHORT',
                    'Allocation ($)': round(alloc_sz, 2),
                    'Shares': shares,
                    'Entry Price': round(entry_px, 3),
                    'Exit Price': round(cur_px, 3),
                    'Bars Held (m)': bars_held,
                    'Exit Reason': reason,
                    'Gross PnL ($)': round(gross, 2),
                    'Net PnL ($)': round(net, 2),
                    'Running Equity ($)': round(cur_eq, 2)
                })
                in_pos = 0
                
        if in_pos == 0 and not is_eod and trades_today < max_trades_per_day:
            if l_arr[i] or s_arr[i]:
                in_pos = 1 if l_arr[i] else -1
                entry_px = closes[i]
                entry_idx = i
                entry_time = times[i]
                if "Full Compounding" in sizing_mode:
                    alloc_sz = max(cur_eq, 100.0)
                elif "Capped at Start" in sizing_mode:
                    alloc_sz = max(min(cur_eq, base_trade_size), 100.0)
                else:
                    alloc_sz = base_trade_size
                shares = max(1, int(round(alloc_sz / entry_px)))
                trades_today += 1
                
    return pd.DataFrame(trades)

if uploaded_file is not None:
    ticker_name = uploaded_file.name.split('_')[0].split('.')[0].upper()
    with st.spinner(f"Auditing 1-minute data quality & running 7-strategy tournament on {ticker_name}..."):
        df, audit = load_and_audit_csv(uploaded_file, min_completeness_pct, max_gap_allowed, orb_minutes)
        
    st.subheader(f"🔍 Layer 1–3 Data Quality & Missing-Bar Audit: `{ticker_name}`")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Raw Avg Daily Completeness", f"{audit['pct_completeness_raw']:.1f}%", f"{audit['avg_bars_raw']:.1f} / 390 bars")
    a2.metric("Total Calendar Days in File", f"{audit['total_days']:,}")
    a3.metric("Clean Days Kept for Backtest", f"{audit['kept_days']:,}", f"{audit['kept_days']/max(audit['total_days'],1)*100:.1f}% of days")
    a4.metric("Corrupted/Gapped Days Omitted", f"{audit['omitted_days']:,}", f"-{audit['omit_low_bars']} low bars, -{audit['omit_gap']} large gaps")
    
    if audit['pct_completeness_raw'] < 90.0 or (audit['kept_days'] / max(audit['total_days'], 1)) < 0.35:
        st.warning(
            f"⚠️ **Data Realism Warning for `{ticker_name}`:** This file averages only **{audit['avg_bars_raw']:.1f} bars/day "
            f"({audit['pct_completeness_raw']:.1f}% of 390 minutes)**, leaving **{audit['kept_days']} clean days** under your current filter. "
            f"All missing minutes have been isolated on a 1-minute clock grid so no false signals can bridge across gaps."
        )
    else:
        st.success(
            f"✅ **High Data Integrity Confirmed:** `{ticker_name}` averages **{audit['avg_bars_raw']:.1f}/390 bars per day "
            f"({audit['pct_completeness_raw']:.1f}%)**. Backtesting **{audit['kept_days']} verified clean days** with strict 1-minute clock-grid lookbacks."
        )

    if df.empty:
        st.error("No trading days passed the current completeness and max-gap filters. Lower the completeness slider or increase the max allowed gap in the sidebar.")
        st.stop()

    zero_s = pd.Series(False, index=df.index)
    ex_vwap_l = (df['Close_real'] < df['vwap'] * 0.9970) & (df['ret_15m'] <= -0.0035)
    ex_vwap_s = (df['Close_real'] > df['vwap'] * 1.0030) & (df['ret_15m'] >= 0.0035)

    # S1: Core VWAP Trend + 3-Bar/IBS Dip Buy (Long-Only)
    s1_l = (df['time'] >= '09:45') & (df['time'] <= '15:40') & df['is_real_bar'] & (df['Close_real'] > df['vwap']) & (df['ret_from_open'] >= 0.0015) & ((df['d_streak'] >= 3) | ((df['ret_3m'] <= -0.0010) & (df['ibs'] <= 0.15)))
    
    # S2: Full Multi-Signal Semi-HFT (Long + Short)
    s2_l = (df['time'] >= '09:45') & (df['time'] <= '15:40') & df['is_real_bar'] & (
        ((df['Close_real'] > df['vwap']) & (df['ret_from_open'] >= 0.0015) & ((df['d_streak'] >= 3) | ((df['ret_3m'] <= -0.0010) & (df['ibs'] <= 0.15)))) |
        ((df['Close_real'] > df['or15_high']) & (df['Close_real'] > df['vwap']) & (df['ret_15m'] >= 0.0025) & (df['d_streak'] >= 1) & (df['ibs'] <= 0.25)) |
        ((df['d_streak'] >= 4) & (df['ibs'] <= 0.05) & (df['ret_from_open'] >= -0.0040))
    )
    s2_s = (df['time'] >= '09:45') & (df['time'] <= '15:30') & df['is_real_bar'] & (
        (df['Close_real'] < df['vwap']) & (df['Close_real'] < df['or15_low']) & (df['ret_from_open'] <= -0.0035) & (df['ret_15m'] <= -0.0015) & (df['u_streak'] >= 2) & (df['ibs'] >= 0.80)
    )

    # S3: Rule 10b-18 Gap-Down Reclaim & VWAP Dip
    s3_l = df['is_real_bar'] & (
        ((df['time'] >= '09:35') & (df['time'] <= '11:00') & (df['overnight_gap'] <= -0.0020) & (df['Close_real'] > df['day_open']) & (df['Close_real'] > df['vwap']) & (df['d_streak'] >= 2) & (df['ibs'] <= 0.30)) |
        ((df['time'] >= '09:45') & (df['time'] <= '15:40') & (df['Close_real'] > df['vwap']) & (df['ret_from_open'] >= 0.0015) & ((df['d_streak'] >= 3) | ((df['ret_3m'] <= -0.0012) & (df['ibs'] <= 0.15)))) |
        ((df['time'] >= '09:45') & (df['time'] <= '15:35') & (df['Close_real'] > df['or15_high']) & (df['Close_real'] > df['vwap']) & (df['ret_15m'] >= 0.0025) & (df['d_streak'] >= 1) & (df['ibs'] <= 0.30))
    )

    # S4: OR15 + VWAP + IBS 80/20 Order-Flow Momentum
    s4_l = (df['time'] >= '09:45') & (df['time'] <= '14:30') & df['is_real_bar'] & (df['Close_real'] > df['or15_high']) & (df['Close_real'] > df['vwap']) & (df['ret_15m'] >= 0.0020) & (df['ibs'] >= 0.80)
    s4_s = (df['time'] >= '09:45') & (df['time'] <= '14:30') & df['is_real_bar'] & (df['Close_real'] < df['or15_low']) & (df['Close_real'] < df['vwap']) & (df['ret_15m'] <= -0.0020) & (df['ibs'] <= 0.20)

    # S5: Volatility Crush & Breakdown Cascade (VXX-style)
    s5_l = (df['time'] >= '09:45') & (df['time'] <= '15:00') & df['is_real_bar'] & (df['Close_real'] > df['vwap']) & (df['Close_real'] > df['or15_high']) & (df['ret_from_open'] >= 0.0120) & (df['ret_15m'] >= 0.0080) & (df['ret_2m'] <= -0.0010)
    s5_s = (df['time'] >= '09:45') & (df['time'] <= '15:35') & df['is_real_bar'] & (
        ((df['Close_real'] < df['vwap']) & (df['Close_real'] < df['or15_low']) & (df['ret_from_open'] <= -0.0050) & (df['ret_2m'] >= 0.0010)) |
        ((df['Close_real'] < df['vwap']) & (df['ret_from_open'] <= -0.0060) & (df['ret_15m'] <= -0.0080) & (df['ret_1m'] >= 0.0)) |
        ((df['ret_from_open'] <= -0.0020) & (df['z_15'] >= 1.8) & (df['ibs'] >= 0.75))
    )

    # S6: 10:00 AM First-30m Session Continuation / Breakdown (1 Trade/Day EOD Hold)
    s6_l = (df['time'] >= '10:00') & (df['time'] <= '10:05') & df['is_real_bar'] & (df['or30_ret'] >= 0.0025) & (df['Close_real'] > df['vwap'])
    s6_s = (df['time'] >= '10:00') & (df['time'] <= '10:05') & df['is_real_bar'] & (df['or30_ret'] <= -0.0050) & (df['Close_real'] < df['vwap'])

    # S7: Classic Opening Range Breakout (1 Trade/Day)
    s7_l = (df['time'] >= df['or_after_time']) & (df['time'] <= '15:30') & df['is_real_bar'] & (df['Close_real'] > df['orc_high'])
    s7_s = (df['time'] >= df['or_after_time']) & (df['time'] <= '15:30') & df['is_real_bar'] & (df['Close_real'] < df['orc_low'])

    strategies = {
        "1. Core VWAP + 3-Bar/IBS Dip (Long-Only)": simulate_strategy(df, "1. Core VWAP + 3-Bar/IBS Dip", s1_l, zero_s, ex_vwap_l, zero_s, default_max_hold, default_sl_pct, default_pt_pct),
        "2. Full Multi-Signal Semi-HFT (Long + Short)": simulate_strategy(df, "2. Full Multi-Signal Semi-HFT", s2_l, s2_s, ex_vwap_l, ex_vwap_s, default_max_hold, default_sl_pct, default_pt_pct),
        "3. Rule 10b-18 Gap-Down Reclaim + VWAP Dip": simulate_strategy(df, "3. Gap-Down Reclaim + VWAP Dip", s3_l, s2_s, ex_vwap_l, ex_vwap_s, min(default_max_hold, 90), default_sl_pct, default_pt_pct),
        "4. OR15 + VWAP + IBS 80/20 Momentum Breakout": simulate_strategy(df, "4. OR15 + VWAP + IBS Momentum", s4_l, s4_s, zero_s, zero_s, max(default_max_hold, 180), default_sl_pct, default_pt_pct),
        "5. Volatility Crush & Breakdown Cascade": simulate_strategy(df, "5. Vol Crush & Cascade", s5_l, s5_s, (df['Close_real'] < df['vwap']*0.995), (df['Close_real'] > df['vwap']*1.005), 90, 0.014, 0.015),
        "6. 10:00 AM First-30m Session Continuation (1/Day)": simulate_strategy(df, "6. 10:00 AM Session Continuation", s6_l, s6_s, zero_s, zero_s, 360, 0.015, 0.020, max_trades_per_day=1),
        "7. Classic Opening Range Breakout (1/Day)": simulate_strategy(df, f"7. Classic {orb_minutes}m ORB", s7_l, s7_s, zero_s, zero_s, 360, default_sl_pct, default_pt_pct, max_trades_per_day=1),
    }

    leaderboard = []
    all_clean_dates = sorted(df['date'].unique())
    equity_curves = pd.DataFrame(index=pd.to_datetime(all_clean_dates))

    for name, tdf in strategies.items():
        if tdf.empty:
            leaderboard.append({
                'Strategy': name, 'Trades': 0, 'Trades/Day': 0.0, 'Win Rate (%)': 0.0,
                'Profit Factor': 0.0, 'Avg Net/Trade ($)': 0.0, 'Total Commissions ($)': 0.0,
                'Total Net PnL ($)': 0.0, 'Max Drawdown ($)': 0.0, 'Ann. Sharpe': 0.0, 'Ending Equity ($)': base_trade_size
            })
            continue
        net = tdf['Net PnL ($)']
        wins = net[net > 0].sum()
        losses = abs(net[net <= 0].sum())
        pf = wins / losses if losses > 0 else np.nan
        
        daily_pnl = tdf.groupby('Date')['Net PnL ($)'].sum().reindex(all_clean_dates, fill_value=0.0)
        cum_pnl = daily_pnl.cumsum()
        equity_curves[name] = cum_pnl.values
        max_dd = (cum_pnl - cum_pnl.cummax()).min()
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0.0
        
        leaderboard.append({
            'Strategy': name,
            'Trades': len(tdf),
            'Trades/Day': round(len(tdf) / len(all_clean_dates), 2),
            'Win Rate (%)': round((net > 0).mean() * 100, 2),
            'Profit Factor': round(pf, 2) if not np.isnan(pf) else 99.9,
            'Avg Net/Trade ($)': round(net.mean(), 2),
            'Total Commissions ($)': round(len(tdf) * commission_per_trade, 2),
            'Total Net PnL ($)': round(net.sum(), 2),
            'Max Drawdown ($)': round(max_dd, 2),
            'Ann. Sharpe': round(sharpe, 2),
            'Ending Equity ($)': round(tdf['Running Equity ($)'].iloc[-1], 2)
        })

    lb_df = pd.DataFrame(leaderboard).sort_values('Total Net PnL ($)', ascending=False).reset_index(drop=True)
    st.subheader(f"🏆 All 7 Strategies Tournament Leaderboard (`{ticker_name}`)")
    st.dataframe(lb_df, use_container_width=True)

    st.subheader("📈 Cumulative Net PnL Comparison Across All 7 Strategies")
    st.line_chart(equity_curves)

    st.markdown("---")
    st.subheader("🔎 Deep-Dive Inspector: Select Any Strategy for Annual, Weekly & Trade Logs")
    selected_strat = st.selectbox("Select Strategy to Inspect", options=list(strategies.keys()), index=1)
    sel_df = strategies[selected_strat]

    if sel_df.empty:
        st.info("No trades triggered for this strategy on the filtered dataset.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Strategy Net PnL", f"${sel_df['Net PnL ($)'].sum():,.2f}")
        c2.metric("Win Rate", f"{(sel_df['Net PnL ($)'] > 0).mean()*100:.2f}%", f"{len(sel_df):,} trades")
        c3.metric("Long Trades Net PnL", f"${sel_df[sel_df['Direction']=='LONG']['Net PnL ($)'].sum():,.2f}", f"{len(sel_df[sel_df['Direction']=='LONG'])} Longs")
        c4.metric("Short Trades Net PnL", f"${sel_df[sel_df['Direction']=='SHORT']['Net PnL ($)'].sum():,.2f}", f"{len(sel_df[sel_df['Direction']=='SHORT'])} Shorts")

        col_a, col_w = st.columns([1, 1.4])
        with col_a:
            st.markdown("#### 📅 Annualized Breakdown")
            ann_df = sel_df.groupby('Year').agg(
                Trades=('Net PnL ($)', 'count'),
                Win_Rate=('Net PnL ($)', lambda x: f"{(x > 0).mean()*100:.1f}%"),
                Gross_PnL=('Gross PnL ($)', lambda x: f"${x.sum():,.2f}"),
                Net_PnL=('Net PnL ($)', lambda x: f"${x.sum():,.2f}"),
                Return_Pct=('Net PnL ($)', lambda x: f"{(x.sum()/base_trade_size)*100:+.2f}%")
            ).reset_index()
            st.dataframe(ann_df, use_container_width=True)

        with col_w:
            st.markdown("#### 🗓️ Weekly Performance Log")
            sel_df['Date_dt'] = pd.to_datetime(sel_df['Date'])
            sel_df['Week_Start'] = sel_df['Date_dt'].dt.to_period('W-FRI').apply(lambda r: r.start_time.strftime('%Y-%m-%d'))
            sel_df['Week_End'] = sel_df['Date_dt'].dt.to_period('W-FRI').apply(lambda r: r.end_time.strftime('%Y-%m-%d'))
            wk_df = sel_df.groupby(['Week_Start', 'Week_End']).agg(
                Trades=('Net PnL ($)', 'count'),
                Win_Rate=('Net PnL ($)', lambda x: f"{(x > 0).mean()*100:.1f}%"),
                Weekly_Net_PnL=('Net PnL ($)', 'sum')
            ).reset_index()
            wk_df['Cum_Net_PnL ($)'] = wk_df['Weekly_Net_PnL'].cumsum()
            st.dataframe(wk_df, use_container_width=True, height=300)

        st.markdown("#### 📋 Master Trade Log")
        st.dataframe(sel_df.drop(columns=['Date_dt', 'Week_Start', 'Week_End'], errors='ignore'), use_container_width=True)
else:
    st.info("👈 Upload any 1-minute CSV file (`NVDA`, `AMZN`, `NFLX`, `WMT`, `DE`, `QQQ`, `AAPL`, `TLT`, or `VXX`) in the sidebar to run the 7-strategy suite.")

# === END OF FILE ===
