import os
import io
import time
import threading
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import socket
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from django.http import HttpResponse
from django.shortcuts import render
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.offline import plot

# Turn off interactive plotting and warnings
plt.ioff()
warnings.filterwarnings("ignore")
plt.close('all')

# --- Global Lock for Plot Generation ---
plot_lock = threading.Lock()

# --- Configuration Constants ---
TIME_INTERVAL = '5min'


# Detect if we are running on server or local
if socket.gethostname() == 'ubuntu-s-1vcpu-512mb-10gb-fra1-01':
    # Server path (adjust if needed)
    TARGET_PATH = '/opt/heatmap_project/data'
else:
    # Local machine path
    TARGET_PATH = r'C:\Users\cuschierii\OneDrive - centralbankmalta.org\FX and Liquidity Management\Ad-hoc Projects\all_currencies_new'

FILE_PATH = os.path.join(TARGET_PATH, f'{TIME_INTERVAL}_core.csv')


# List of available currencies (should match your CSV data)
CURRENCIES = ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURNZD', 'EURCAD', 'EURNOK', 'EURSEK']
DAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
HOURS = list(range(24))  # assuming hourly buckets for 0-23

# --- Default Date Range (using full datetime now) ---
DEFAULT_START_DATE = pd.to_datetime("2023-04-13")
DEFAULT_END_DATE = pd.to_datetime(datetime.today().strftime("%Y-%m-%d %H:%M"))
# -----------------------------------------------------------------------------------
#                           HELPER / SHARED FUNCTIONS
# -----------------------------------------------------------------------------------

def get_date_param(request, param_name, default_value):
    """
    Extract a datetime parameter from GET; fallback to default_value if not present.
    Supports both "YYYY-MM-DD" and "YYYY-MM-DDTHH:MM" formats.
    """
    date_str = request.GET.get(param_name, None)
    if date_str:
        # Replace "T" with space if using datetime-local input
        date_str = date_str.replace("T", " ")
        try:
            return pd.to_datetime(date_str)
        except Exception as e:
            return default_value
    return default_value

def load_data(currency, start_date=None, end_date=None):
    """
    Load the main CSV data, filter by weekday, then extract and return:
      - data for the specified currency (bid)
      - an hourly pivot for counting returns (hourly)
    """
    df = pd.read_csv(FILE_PATH, parse_dates=['Time'])
    df = df.set_index('Time')

    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    # Weekdays only
    df = df[df.index.dayofweek < 5]

    bid_cols = [col for col in df.columns if 'Bid' in col]
    # In your CSV, these columns order matches the order of CURRENCIES, so rename:
    bid_df = df[bid_cols].copy()
    bid_df.columns = CURRENCIES

    data = bid_df[[currency]].copy()

    # Compute hourly returns (first vs last price in that hour)
    hourly = data.resample('H', closed='left', label='left').agg(['first', 'last'])
    hourly.columns = ['first_price', 'last_price']
    hourly['hourly_return'] = (hourly['last_price'] - hourly['first_price']) / hourly['first_price'] * 100
    hourly.dropna(inplace=True)
    hourly['day'] = hourly.index.dayofweek
    hourly['hour'] = hourly.index.hour
    return data, hourly

# -----------------------------------------------------------------------------------
#                                 HEATMAP VIEWS
# -----------------------------------------------------------------------------------

def index(request):
    """
    Main Heatmap index page with form for currency and date selection.
    Renders the large clickable heatmap (and the tabular day/hour selection).
    """
    selected_currency = request.GET.get('currency', 'EURGBP')

    # Get user-selected start/end date or default
    start_date = get_date_param(request, 'start_date', DEFAULT_START_DATE)
    end_date = get_date_param(request, 'end_date', DEFAULT_END_DATE)

    # Build HTML for clickable image map
    x0, y0 = 900, 600  # top-left corner of clickable region (pixels)
    cell_width = 900 / 24
    cell_height = 600 / 5

    image_map_html = '<map name="heatmapmap">'
    for day_index, day in enumerate(DAY_LABELS):
        for hour in range(24):
            left = int(x0 + hour * cell_width)
            top = int(y0 + day_index * cell_height)
            right = int(left + cell_width)
            bottom = int(top + cell_height)
            url = (f"/detail/?currency={selected_currency}&selected_day={day}&selected_hour={hour}"
                   f"&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}")
            image_map_html += f'<area shape="rect" coords="{left},{top},{right},{bottom}" href="{url}" alt="{day} {hour}">'
    image_map_html += '</map>'

    context = {
        'selected_currency': selected_currency,
        'currencies': CURRENCIES,
        'days': DAY_LABELS,
        'hours': HOURS,
        'image_map': image_map_html,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
    }
    return render(request, 'heatmap_app/index.html', context)


def heatmap_image(request):
    """
    Returns the 2x2 heatmap image as PNG (Wins, Losses, Neutrals, Total Trades).
    """
    currency = request.GET.get('currency', 'EURGBP')
    sel_day = request.GET.get('selected_day')
    sel_hour = request.GET.get('selected_hour')

    start_date = get_date_param(request, 'start_date', DEFAULT_START_DATE)
    end_date = get_date_param(request, 'end_date', DEFAULT_END_DATE)

    sel_day_int = DAY_LABELS.index(sel_day) if sel_day in DAY_LABELS else None
    sel_hour = int(sel_hour) if sel_hour is not None else None

    with plot_lock:
        _, hourly = load_data(currency, start_date, end_date)

        # Helper to reorder days so it's always Mon-Fri in correct order
        def reorder_days(df):
            return df.reindex([0, 1, 2, 3, 4])  # Mon(0) to Fri(4)

        wins = reorder_days(hourly[hourly['hourly_return'] > 0]
                            .pivot_table(values='hourly_return', index='day', columns='hour', aggfunc='count')
                            .fillna(0))
        losses = reorder_days(hourly[hourly['hourly_return'] < 0]
                              .pivot_table(values='hourly_return', index='day', columns='hour', aggfunc='count')
                              .fillna(0))
        neutrals = reorder_days(hourly[hourly['hourly_return'] == 0]
                                .pivot_table(values='hourly_return', index='day', columns='hour', aggfunc='count')
                                .fillna(0))
        entries = reorder_days(hourly
                               .pivot_table(values='hourly_return', index='day', columns='hour', aggfunc='count')
                               .fillna(0))

        # Apply dark background
        plt.style.use('dark_background')
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))

        heatmap_specs = [
            (wins, f'Wins (Positive Returns) for {currency}', 'Greens'),
            (losses, f'Losses (Negative Returns) for {currency}', 'Reds'),
            (neutrals, f'Neutrals (0 Returns) for {currency}', 'Oranges'),
            (entries, f'Total Trades per Hour for {currency}', 'viridis_r')
        ]

        def annotate(ax, data, im, total_data=None):
            cmap = im.get_cmap()
            norm = im.norm
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    value = data.iloc[i, j]
                    rgba = cmap(norm(value))  # RGBA color
                    brightness = (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2])
                    text_color = "white" if brightness < 0.5 else "black"

                    if total_data is not None:
                        total = total_data.iloc[i, j]
                        perc = (value / total * 100) if total != 0 else 0
                        ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                                fontsize=10, color=text_color, fontweight='bold')
                        ax.text(j, i + 0.3, f"{perc:.0f}%", ha="center", va="center",
                                fontsize=8, fontstyle="italic", color=text_color)
                    else:
                        ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                                fontsize=10, color=text_color)

        for ax, (table, title, cmap_name) in zip(axs.flatten(), heatmap_specs):
            im = ax.imshow(table, aspect='auto', origin='upper', cmap=cmap_name)
            ax.set_title(title, color='white')
            ax.set_xlabel('Hour of Day', color='white')
            ax.set_xticks(np.arange(len(table.columns)))
            ax.set_xticklabels(table.columns, color='white')
            ax.set_yticks(np.arange(len(table.index)))
            ax.set_yticklabels([DAY_LABELS[int(d)] for d in table.index], color='white')

            if "Wins" in title or "Losses" in title:
                annotate(ax, table, im, total_data=entries)
            else:
                annotate(ax, table, im)

            # Highlight user selection
            if sel_hour is not None and sel_day_int is not None:
                if sel_hour in table.columns and sel_day_int in table.index:
                    x = list(table.columns).index(sel_hour)
                    y = list(table.index).index(sel_day_int)
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                           fill=False, edgecolor='blue', linewidth=2))

            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


def detail(request):
    """
    Detail page showing two plots for the selected currency/day/hour.
    """
    currency = request.GET.get('currency', 'EURGBP')
    selected_day = request.GET.get('selected_day', 'Thu')
    selected_hour = request.GET.get('selected_hour', '20')
    start_date = request.GET.get('start_date', DEFAULT_START_DATE.strftime('%Y-%m-%d'))
    end_date = request.GET.get('end_date', DEFAULT_END_DATE.strftime('%Y-%m-%d'))

    context = {
        'selected_currency': currency,
        'selected_day': selected_day,
        'selected_hour': selected_hour,
        'start_date': start_date,
        'end_date': end_date,
        'timestamp': int(time.time()),
    }
    return render(request, 'heatmap_app/detail.html', context)


def historical_plot_image(request):
    """
    Returns a plot focusing on the selected day/hour price data over time.
    """
    currency = request.GET.get('currency', 'EURGBP')
    selected_day = request.GET.get('selected_day', 'Thu')
    selected_hour = int(request.GET.get('selected_hour', '20'))

    start_date = get_date_param(request, 'start_date', DEFAULT_START_DATE)
    end_date = get_date_param(request, 'end_date', DEFAULT_END_DATE)

    day_to_int = {d: i for i, d in enumerate(DAY_LABELS)}
    sel_day_int = day_to_int.get(selected_day, 3)

    with plot_lock:
        data, hourly = load_data(currency, start_date, end_date)
        buckets = hourly[(hourly['day'] == sel_day_int) & (hourly['hour'] == selected_hour)]

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(data.index, data[currency], label=f'{currency} Price', color='cyan', lw=0.7)

        if buckets.empty:
            ax.text(0.5, 0.5, f"No data for {selected_day} at {selected_hour}H.",
                    transform=ax.transAxes, ha='center', va='center', fontsize=16)
        else:
            gain_label, loss_label, neutral_label = False, False, False
            for ts, row in buckets.iterrows():
                bucket_end = ts + pd.Timedelta(hours=1)
                bucket_data = data[(data.index >= ts) & (data.index < bucket_end)]
                if bucket_data.empty:
                    continue
                ret = row['hourly_return']
                if ret > 0:
                    color = 'green'
                    label = f'{selected_day} {selected_hour}H (Gain)' if not gain_label else None
                    gain_label = True
                elif ret < 0:
                    color = 'red'
                    label = f'{selected_day} {selected_hour}H (Loss)' if not loss_label else None
                    loss_label = True
                else:
                    color = 'orange'
                    label = f'{selected_day} {selected_hour}H (Neutral)' if not neutral_label else None
                    neutral_label = True
                ax.axvspan(ts, bucket_end, color=color, alpha=0.7, label=label)

        ax.set_title(f'{currency} Price with {selected_day} {selected_hour}H Buckets Highlighted')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


def line_plot_image(request):
    """
    Returns a plot that aligns the selected day/hour price paths to see changes side-by-side.
    """
    currency = request.GET.get('currency', 'EURGBP')
    selected_day = request.GET.get('selected_day', 'Thu')
    selected_hour = int(request.GET.get('selected_hour', '20'))

    start_date = get_date_param(request, 'start_date', DEFAULT_START_DATE)
    end_date = get_date_param(request, 'end_date', DEFAULT_END_DATE)

    day_to_int = {d: i for i, d in enumerate(DAY_LABELS)}
    sel_day_int = day_to_int.get(selected_day, 3)

    with plot_lock:
        data, hourly = load_data(currency, start_date, end_date)
        buckets = hourly[(hourly['day'] == sel_day_int) & (hourly['hour'] == selected_hour)]
        buckets = buckets.sort_index()

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(15, 14), sharex=True)

        for ax in axes:
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # 1) Chronological Coloring
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(0, len(buckets) - 1)
        ax = axes[0]
        for i, (ts, row) in enumerate(buckets.iterrows()):
            bucket_end = ts + pd.Timedelta(hours=1)
            bucket_data = data[(data.index >= ts) & (data.index < bucket_end)]
            if bucket_data.empty:
                continue
            start_price = bucket_data.iloc[0][currency]
            pct_change = (bucket_data[currency] - start_price) / start_price * 100
            pct_change.iloc[-1] = row['hourly_return']  # ensure last matches the known final
            rel_time = (bucket_data.index - ts).total_seconds() / 60
            ax.plot(rel_time, pct_change, color=cmap(norm(i)), lw=1)
        ax.axhline(0, color='white', linewidth=2)
        ax.set_title(f'Aligned Percentage Price Paths: {currency}, {selected_day} @ {selected_hour}H (Chronological)')
        ax.set_ylabel('Percentage Change (%)')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Chronological Order (Bucket Index)',
                     orientation="vertical", fraction=0.02, pad=0.02)

        # 2) Return Sign Coloring
        ax = axes[1]
        for ts, row in buckets.iterrows():
            bucket_end = ts + pd.Timedelta(hours=1)
            bucket_data = data[(data.index >= ts) & (data.index < bucket_end)]
            if bucket_data.empty:
                continue
            start_price = bucket_data.iloc[0][currency]
            pct_change = (bucket_data[currency] - start_price) / start_price * 100
            pct_change.iloc[-1] = row['hourly_return']
            rel_time = (bucket_data.index - ts).total_seconds() / 60
            if row['hourly_return'] > 0:
                color = 'lime'
            elif row['hourly_return'] < 0:
                color = 'red'
            else:
                color = 'orange'
            ax.plot(rel_time, pct_change, color=color, lw=1)

        ax.axhline(0, color='white', linewidth=2)
        ax.set_title(f'Aligned Percentage Price Paths: {currency}, {selected_day} @ {selected_hour}H (By Return Sign)')
        ax.set_xlabel('Minutes from Bucket Start')
        ax.set_ylabel('Percentage Change (%)')

        discrete_colors = ['red', 'orange', 'lime']
        discrete_labels = ['Negative', 'Neutral', 'Positive']
        discrete_cmap = ListedColormap(discrete_colors)
        discrete_norm = BoundaryNorm([-1, 0, 1, 2], discrete_cmap.N)
        sm2 = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=discrete_norm)
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
        cbar2.set_ticks([-0.5, 0.5, 1.5])
        cbar2.set_ticklabels(discrete_labels)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

# -----------------------------------------------------------------------------------
#                          NEW: REBASING VIEWS (Relative Value)
# -----------------------------------------------------------------------------------

def get_rebased_data(start_date, end_date):
    """
    Load the entire CSV, filter to weekdays, then slice by start_date and end_date.
    Return a dataframe of rebased % changes for all EUR* pairs (BID).
    """
    df = pd.read_csv(FILE_PATH)
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)

    # Keep only weekdays
    df = df[df.index.dayofweek < 5]

    # Slice by selected date range
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    # Identify columns
    columns = ['EURUSD','EURGBP','EURJPY','EURCHF','EURAUD','EURNZD','EURCAD','EURNOK','EURSEK']
    bid_columns = [col for col in df.columns if 'Bid' in col]

    # We'll just use the bid columns for the rebasing
    bid_df = df[bid_columns].copy()
    bid_df.columns = columns

    if len(bid_df) == 0:
        return pd.DataFrame(), columns

    # Rebase: (price / first_price - 1) * 100 using the earliest timestamp
    rebased = (bid_df / bid_df.iloc[0] - 1) * 100
    return rebased, columns


def rebasing(request):
    # Use full datetime for defaults (with minute resolution)
    now = datetime.now()
    default_start = now - timedelta(days=30)

    # Get start_date and end_date from GET parameters; if missing, use defaults.
    start_date = get_date_param(request, 'start_date', default_start)
    end_date = get_date_param(request, 'end_date', now)

    # Format timestamps for input fields as "YYYY-MM-DDTHH:MM"
    fmt = "%Y-%m-%dT%H:%M"
    context = {
        'start_date': start_date.strftime(fmt),
        'end_date': end_date.strftime(fmt),
        'timestamp': int(time.time()),
        'today': now.strftime(fmt),
        'one_day': (now - timedelta(days=1)).strftime(fmt),
        'one_week': (now - timedelta(days=7)).strftime(fmt),
        'one_month': (now - timedelta(days=30)).strftime(fmt),
        'one_year': (now - timedelta(days=365)).strftime(fmt),
    }

    # Get the rebased data for the selected date range.
    rebased, columns = get_rebased_data(start_date, end_date)

    # Create a Plotly figure with a dark theme.
    fig = go.Figure()

    # Define a set of bold, high-contrast colors
    bold_colors = ["#FF8C00", "#FF0000", "#7f05e3", "#FFFF00", "#0000FF", "#00FFFF", "#FFFFFF", "#FF69B4", "#32CD32"]

    if not rebased.empty:
        for i, col in enumerate(columns):
            fig.add_trace(go.Scatter(
                x=rebased.index,
                y=rebased[col],
                mode='lines',
                name=col,
                line=dict(color=bold_colors[i % len(bold_colors)], width=2),
                connectgaps=False
            ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="% Change from Start Date",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            width=1400,   
            height=800,  
            margin=dict(t=30, b=50),
        )
        
        # Skip weekends on the x-axis
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])] )
    else:
        fig.add_annotation(
            text="No data available for the selected date range.",
            showarrow=False,
            font=dict(color="red", size=16)
        )

    # Generate the Plotly chart as HTML
    plot_div = plot(fig, output_type="div", include_plotlyjs=True)
    context['plot_div'] = plot_div

    return render(request, 'heatmap_app/rebasing.html', context)


def rebasing_image(request):
    """
    Returns a PNG line chart of rebased % changes for all EUR-based currency pairs,
    given start_date and end_date from GET params.
    """
    now = datetime.now()
    default_start = now - timedelta(days=30)

    start_date = get_date_param(request, 'start_date', default_start)
    end_date = get_date_param(request, 'end_date', now)

    with plot_lock:
        rebased, columns = get_rebased_data(start_date, end_date)

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.style.use('default')  # or 'dark_background'

        if rebased.empty:
            ax.text(0.5, 0.5, "No data available for the selected date range.",
                    transform=ax.transAxes, ha='center', va='center', fontsize=14, color='red')
        else:
            for col in columns:
                ax.plot(rebased.index, rebased[col], label=col, linewidth=1)

            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax.set_xlabel("Time")
            ax.set_ylabel("% Change from Start Date")
            start_str = start_date.strftime("%Y-%m-%d %H:%M")
            end_str = end_date.strftime("%Y-%m-%d %H:%M")
            ax.set_title(f"Rebased EUR Currencies ({start_str} to {end_str})")
            ax.legend(loc='best', ncol=3, fontsize=8)
            plt.xticks(rotation=45)

            plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response
