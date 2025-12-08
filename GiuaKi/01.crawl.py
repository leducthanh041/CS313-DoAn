# -*- coding: utf-8 -*-
"""
Crawl EVN hồ chứa thủy điện:
- Thời gian: 01/01/2022 -> 31/12/2024
- Khung giờ: 02:00, 05:00, 08:00, 11:00, 14:00, 17:00, 20:00, 23:00
- GET: td=dd/MM/yyyy HH:mm & vm=& lv=& hc=2-3-4-76-77
- Làm sạch nhẹ (cắt 'đồng bộ lúc...', xóa header vùng, thêm năm cho 'dd/MM HH:mm').
- Theo dõi tiến độ bằng tqdm.
- KHÔNG thêm cột td_query vào CSV.
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
OUT_CSV = "evn_thuydien_2022_2024.csv"
HC_PARAM = "2-3-4-76-77"
HOURS_PER_DAY = [2, 5, 8, 11, 14, 17, 20, 23]

# ============ Regex & helpers ============
RE_DDMM_HHMM = re.compile(r"\b(?P<dm>\d{2}/\d{2})\s+(?P<hm>\d{2}:\d{2})\b")
RE_CUT_SYNC_TAIL = re.compile(r"(?i)\s*đồng\s*bộ\s*lúc:.*$")

def normalize_ws(x):
    if not isinstance(x, str):
        return x
    return " ".join(x.split()).strip()

def add_year_to_cell(value: str, year: int) -> str:
    if not isinstance(value, str) or not value:
        return value
    value = normalize_ws(value)
    def _dmhm(m):  # dd/MM HH:mm -> dd/MM/yyyy HH:mm
        return f"{m.group('dm')}/{year} {m.group('hm')}"
    return RE_DDMM_HHMM.sub(_dmhm, value)

def cut_sync_tail(value: str) -> str:
    if not isinstance(value, str) or not value:
        return value
    value = normalize_ws(value)
    value = RE_CUT_SYNC_TAIL.sub("", value)
    return value.strip(" -–—|").strip()

def drop_region_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Loại các dòng header vùng (vd. toàn 'Tây Bắc Bộ')."""
    def is_region_row(row: pd.Series) -> bool:
        vals = []
        for v in row.values:
            if isinstance(v, str):
                s = normalize_ws(v)
                if s:
                    vals.append(s)
        return len(vals) >= 2 and len(set(vals)) == 1
    mask = df.apply(is_region_row, axis=1)
    return df[~mask].reset_index(drop=True)

def drop_note_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Xoá dòng 'Chú thích ký hiệu' nếu còn."""
    mask = df.astype(str).apply(
        lambda col: col.str.fullmatch(r"(?i)\s*chú thích ký hiệu\s*"), axis=0
    ).any(axis=1)
    return df[~mask].reset_index(drop=True)

def clean_dataframe(df: pd.DataFrame, year: int) -> pd.DataFrame:
    # thêm năm + cắt 'Đồng bộ lúc: ...'
    df = df.applymap(lambda x: add_year_to_cell(x, year) if isinstance(x, str) else x)
    df = df.applymap(lambda x: cut_sync_tail(x) if isinstance(x, str) else x)
    # bỏ header vùng + chú thích
    df = drop_region_header_rows(df)
    df = drop_note_rows(df)
    return df

# ============ HTTP ============
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

# ============ Parse ============
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
    df_all = clean_dataframe(df_all, dt.year)
    df_all = df_all.drop_duplicates()  # khử trùng trong 1 mốc (giữ nguyên kiểu dữ liệu)
    # KHÔNG thêm cột 'td_query' vào dữ liệu
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
    seen_td = set()  # tránh gọi trùng cùng 1 mốc thời gian

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

            # Ghi tạm theo lô
            if len(batch) >= flush_every:
                combined = pd.concat(batch, ignore_index=True, sort=False)
                # KHÔNG drop_duplicates ở cấp batch để không xoá nhầm các dòng giống nhau giữa các mốc khác nhau
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

# ============ Run ============
if __name__ == "__main__":
    crawl_range_to_csv(
        start_date="01/01/2022",
        end_date="17/10/2025",
        hours=[2,5,8,11,14,17,20,23],
        hc="2-3-4-76-77",
        out_csv="data_thuydien.csv",
        flush_every=120,
        polite_delay=0,
        overwrite=True
    )
