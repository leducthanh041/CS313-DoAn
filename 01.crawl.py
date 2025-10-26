# -*- coding: utf-8 -*-
"""
Crawl EVN hồ chứa thủy điện (fix lỗi 'DataFrame' object has no attribute 'str'):
- Ghi đè chỉ cột thời gian về mốc crawl
- Xóa header vùng, cắt 'đồng bộ lúc...' trong cột "Tên hồ"
- tqdm, không thêm td_query
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Iterable, Optional
from tqdm.auto import tqdm

BASE_URL = "https://hochuathuydien.evn.com.vn/PageHoChuaThuyDienEmbedEVN.aspx"
OUT_CSV = "data_thuydien.csv"
HC_PARAM = "3"  # hoặc "2-3-4-76-77"
HOURS_PER_DAY = [2, 5, 8, 11, 14, 17, 20, 23]

# Regex thời gian: dd/MM[/yyyy] HH:mm
RE_ANY_TS = re.compile(r"\b(?P<dd>\d{2})/(?P<mm>\d{2})(?:/(?P<yyyy>\d{4}))?\s+(?P<hh>\d{2}):(?P<min>\d{2})\b")
# Regex cắt "đồng bộ lúc ..."
RE_CUT_SYNC_TAIL = re.compile(r"(?i)\s*đồng\s*bộ\s*lúc:.*$")

# Gợi ý tên cột thời gian hay gặp
TIME_COL_NAME_HINTS = {"thời điểm", "thoi diem", "thoi_điểm", "thoi_diem", "thoidiem", "thời_điểm"}

def normalize_ws(x):
    if pd.isna(x):
        return ""
    if not isinstance(x, str):
        x = str(x)
    return " ".join(x.split()).strip()

def override_ts_to_base_in_series(s: pd.Series, base_dt: datetime) -> pd.Series:
    """Chỉ thay thời gian trong Series (đã xác định là cột thời gian) sang base_dt."""
    target = base_dt.strftime("%d/%m/%Y %H:%M")
    # replace bằng regex trên toàn bộ Series (an toàn hơn .str nếu có NaN)
    return s.astype(str).replace(RE_ANY_TS, target, regex=True).map(lambda v: normalize_ws(v))

def detect_time_columns(df: pd.DataFrame) -> List[str]:
    """Tự phát hiện cột thời gian:
       - Ưu tiên tên cột có gợi ý (time hints)
       - Hoặc tỉ lệ giá trị khớp pattern thời gian >= 0.5"""
    time_cols = []
    for col in df.columns:
        col_lc = str(col).strip().lower()
        if any(hint in col_lc for hint in TIME_COL_NAME_HINTS):
            time_cols.append(col)
            continue
        series = df[col].astype(str).fillna("")
        # tỉ lệ giá trị khớp pattern
        matches = series.str.contains(RE_ANY_TS)
        # nếu series toàn "" sẽ trả False => match_ratio 0
        match_ratio = matches.mean() if len(matches) > 0 else 0.0
        if match_ratio >= 0.5:
            time_cols.append(col)
    return time_cols

def cut_sync_tail_in_tenho_series(s: pd.Series) -> pd.Series:
    """Cắt 'đồng bộ lúc...' trong Series (chỉ dùng cho cột 'Tên hồ')."""
    # dùng replace regex trực tiếp (an toàn với NaN)
    return s.astype(str).replace(RE_CUT_SYNC_TAIL, "", regex=True).map(lambda v: normalize_ws(v)).map(lambda v: v.strip(" -–—|"))

def drop_region_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Loại dòng header vùng (vd. 'Tây Bắc Bộ'): nhiều ô text không rỗng nhưng tất cả giống nhau."""
    def is_region_row(row: pd.Series) -> bool:
        vals = []
        for v in row.values:
            if pd.isna(v):
                continue
            t = normalize_ws(v)
            if t:
                vals.append(t)
        return len(vals) >= 2 and len(set(vals)) == 1
    mask = df.apply(is_region_row, axis=1)
    return df.loc[~mask].reset_index(drop=True)

def drop_note_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Xoá dòng 'Chú thích ký hiệu' nếu còn."""
    # Tạo DataFrame boolean: từng ô có khớp exact phrase "chú thích ký hiệu" (case-insensitive) không
    df_str = df.fillna("").astype(str).applymap(lambda v: v.strip().lower())
    mask_row = df_str.apply(lambda row: any(cell == "chú thích ký hiệu" for cell in row), axis=1)
    return df.loc[~mask_row].reset_index(drop=True)

def clean_dataframe_only_time_cols(df: pd.DataFrame, base_dt: datetime) -> pd.DataFrame:
    # 1) CHỈ ghi đè ở các cột thời gian (phát hiện tự động)
    time_cols = detect_time_columns(df)
    for c in time_cols:
        df[c] = override_ts_to_base_in_series(df[c], base_dt)

    # 2) Cắt 'đồng bộ lúc...' CHỈ trong cột "Tên hồ"
    if "Tên hồ" in df.columns:
        df["Tên hồ"] = cut_sync_tail_in_tenho_series(df["Tên hồ"])

    # 3) Xóa dòng header vùng + chú thích
    df = drop_region_header_rows(df)
    df = drop_note_rows(df)
    return df

# ================= HTTP =================
def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    return s

def fetch_html(session: requests.Session, date_time_str: str, hc: str, pbar: tqdm) -> str:
    params = {"td": date_time_str, "vm": "", "lv": "", "hc": hc}
    for attempt in range(4):
        try:
            r = session.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            pbar.write(f"[GET] {r.url}")
            if attempt > 0:
                pbar.write(f"[RETRY OK] {date_time_str}")
            return r.text
        except Exception as e:
            wait = 2 * (attempt + 1)
            pbar.write(f"[WARN] {e} khi GET td={date_time_str}. Thử lại sau {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"GET thất bại sau retry: td={date_time_str}")

# ================= Parse =================
def parse_all_tables(html: str) -> List[pd.DataFrame]:
    try:
        dfs = pd.read_html(html)
        if dfs:
            return dfs
    except ValueError:
        pass

    soup = BeautifulSoup(html, "html.parser")
    dfs: List[pd.DataFrame] = []
    for tb in soup.find_all("table"):
        rows = []
        for tr in tb.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if rows:
            try:
                if len(rows) > 1 and len(set(map(len, rows))) == 1:
                    header, data = rows[0], rows[1:]
                    dfs.append(pd.DataFrame(data, columns=header))
                else:
                    dfs.append(pd.DataFrame(rows))
            except Exception:
                continue
    return dfs

def crawl_one_timestamp(session: requests.Session, dt: datetime, hc: str, pbar: tqdm) -> Optional[pd.DataFrame]:
    td_str = dt.strftime("%d/%m/%Y %H:00")
    html = fetch_html(session, td_str, hc, pbar)
    dfs = parse_all_tables(html)
    if not dfs:
        pbar.write(f"[INFO] Không thấy bảng: {td_str}")
        return None
    df_all = pd.concat(dfs, ignore_index=True, sort=False)

    # Chỉ ghi đè cột thời gian về đúng mốc dt + làm sạch nhẹ
    df_all = clean_dataframe_only_time_cols(df_all, dt)

    # Khử trùng trong 1 mốc (một số bảng có thể lặp dòng)
    df_all = df_all.drop_duplicates()

    return df_all

# ============ Lập lịch thời điểm ============
def generate_datetimes(start_date: datetime, end_date: datetime, hours: Iterable[int]) -> List[datetime]:
    cur = datetime(start_date.year, start_date.month, start_date.day)
    end = datetime(end_date.year, end_date.month, end_date.day)
    out = []
    while cur <= end:
        for h in hours:
            out.append(cur.replace(hour=h, minute=0, second=0, microsecond=0))
        cur += timedelta(days=1)
    return out

# ============ Pipeline tổng (tqdm) ============
def crawl_range_to_csv(
    start_date: str = "01/01/2022",
    end_date: str = "31/12/2024",
    hours: Iterable[int] = HOURS_PER_DAY,
    hc: str = HC_PARAM,
    out_csv: str = OUT_CSV,
    flush_every: int = 120,
    polite_delay: float = 0.8,
    overwrite: bool = True
):
    sess = requests_session()
    sd = datetime.strptime(start_date, "%d/%m/%Y")
    ed = datetime.strptime(end_date, "%d/%m/%Y")
    dts = generate_datetimes(sd, ed, hours)

    out_path = Path(out_csv)
    if overwrite and out_path.exists():
        out_path.unlink()

    batch = []
    seen_td = set()  # bảo vệ gọi trùng

    with tqdm(total=len(dts), desc="Crawling EVN", unit="ts") as pbar:
        for i, dt in enumerate(dts, 1):
            td_key = dt.strftime("%d/%m/%Y %H:00")
            if td_key in seen_td:
                pbar.write(f"[SKIP] Trùng mốc {td_key}")
                pbar.update(1)
                continue

            try:
                df = crawl_one_timestamp(sess, dt, hc, pbar)
                if df is not None and not df.empty:
                    batch.append(df)
                    seen_td.add(td_key)
                    pbar.set_postfix({"batch": len(batch)})
                else:
                    pbar.set_postfix({"batch": len(batch)})
            except Exception as e:
                pbar.write(f"[ERROR] {td_key} -> {e}")

            # Ghi theo lô
            if len(batch) >= flush_every:
                combined = pd.concat(batch, ignore_index=True, sort=False)
                if out_path.exists():
                    combined.to_csv(out_path, mode="a", header=False, index=False, encoding="utf-8-sig")
                else:
                    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
                pbar.write(f"[FLUSH] Ghi {len(combined)} dòng. Tiến độ {i}/{len(dts)}.")
                batch = []

            if polite_delay > 0:
                time.sleep(polite_delay)

            pbar.update(1)

    # Ghi phần còn lại
    if batch:
        combined = pd.concat(batch, ignore_index=True, sort=False)
        if out_path.exists():
            combined.to_csv(out_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] Hoàn tất crawl. File: {out_path.resolve()}")

# ============ Run (GỢI Ý TEST) ============
if __name__ == "__main__":
    crawl_range_to_csv(
        start_date="1/9/2025",
        end_date="24/10/2025",
        hours=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],                 # ví dụ test 00:00
        hc=HC_PARAM,
        out_csv="data_thuydien.csv",
        flush_every=1,             # ghi ngay mỗi mốc để dễ kiểm tra
        polite_delay=0.0,
        overwrite=True
    )
