# -*- coding: utf-8 -*-
"""
Crawl EVN hồ chứa thủy điện:
- Thời gian: 01/01/2022 -> 31/12/2024 (tùy start/end)
- Khung giờ: list HOURS_PER_DAY
- GET: td=dd/MM/yyyy HH:00 & vm=& lv=& hc=HC_PARAM
- Behavior quan trọng:
  * Nếu server trả dữ liệu của thời điểm trước đó, vẫn dùng dữ liệu đó
    nhưng CHỈ ghi đè các ô thời gian (cột timestamp) về đúng mốc đang crawl.
  * In log khi phát hiện server trả timestamp khác mốc yêu cầu.
- Cleanup nhẹ: xóa header vùng, cắt 'Đồng bộ lúc...' trong cột "Tên hồ".
- Không thêm cột td_query vào CSV.
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
HC_PARAM = "2-3-4-76-77"
HOURS_PER_DAY = [2, 5, 8, 11, 14, 17, 20, 23]

# Regex: dd/MM HH:mm  (trang thường hiển thị không có năm)
RE_DDMM_HHMM = re.compile(r"\b(?P<dm>\d{2}/\d{2})\s+(?P<hm>\d{2}:\d{2})\b")
RE_CUT_SYNC_TAIL = re.compile(r"(?i)\s*đồng\s*bộ\s*lúc:.*$")

# Gợi ý tên cột thời gian
TIME_COL_NAME_HINTS = {"thời điểm", "thoi diem", "thoi_điểm", "thoi_diem", "thoidiem", "time", "timestamp"}

def normalize_ws(x):
    if pd.isna(x):
        return ""
    if not isinstance(x, str):
        x = str(x)
    return " ".join(x.split()).strip()

def detect_time_columns(df: pd.DataFrame) -> List[str]:
    """Tự phát hiện cột thời gian:
       - ưu tiên tên cột có hint,
       - nếu không, đánh giá theo tỉ lệ ô khớp RE_DDMM_HHMM >= 0.5
    """
    time_cols = []
    for col in df.columns:
        col_lc = str(col).strip().lower()
        if any(hint in col_lc for hint in TIME_COL_NAME_HINTS):
            time_cols.append(col)
            continue
        ser = df[col].astype(str).fillna("")
        matches = ser.str.contains(RE_DDMM_HHMM, regex=True)
        if len(matches) > 0 and matches.mean() >= 0.5:
            time_cols.append(col)
    return time_cols

def find_first_original_timestamps(df: pd.DataFrame, max_show: int = 5) -> List[str]:
    """Tìm các timestamp đầu tiên xuất hiện trong bảng (dd/MM HH:MM)."""
    found = []
    for col in df.columns:
        for v in df[col].astype(str).fillna(""):
            m = RE_DDMM_HHMM.search(v)
            if m:
                found.append(f"{m.group('dm')} {m.group('hm')}")
                if len(found) >= max_show:
                    return list(dict.fromkeys(found))
    return list(dict.fromkeys(found))

def cut_sync_tail_in_tenho_series(s: pd.Series) -> pd.Series:
    return s.astype(str).replace(RE_CUT_SYNC_TAIL, "", regex=True).map(lambda v: normalize_ws(v)).map(lambda v: v.strip(" -–—|"))

def drop_region_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    def is_region_row(row: pd.Series) -> bool:
        vals = []
        for v in row.values:
            if pd.isna(v):
                continue
            t = normalize_ws(v)
            if t:
                vals.append(t)
        # row region thường có >=2 ô text và tất cả giống nhau
        return len(vals) >= 2 and len(set(vals)) == 1
    mask = df.apply(is_region_row, axis=1)
    return df.loc[~mask].reset_index(drop=True)

def drop_note_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_s = df.fillna("").astype(str).applymap(lambda v: v.strip().lower())
    mask = df_s.apply(lambda row: any(cell == "chú thích ký hiệu" for cell in row), axis=1)
    return df.loc[~mask].reset_index(drop=True)

def clean_and_override_time_columns(df: pd.DataFrame, base_dt: datetime, pbar: tqdm) -> pd.DataFrame:
    """
    - trước tiên: phát hiện timestamp gốc (nếu có) và in log khi khác base_dt
    - phát hiện cột thời gian
    - CHỈ ghi đè các ô khớp pattern dd/MM HH:MM trong các cột thời gian thành base_dt (dd/MM/YYYY HH:MM)
    - cắt 'Đồng bộ lúc...' chỉ ở cột 'Tên hồ'
    - loại header vùng & note rows
    """
    # tìm một vài timestamp gốc để kiểm tra (trước khi ghi đè)
    orig_ts = find_first_original_timestamps(df, max_show=5)
    if orig_ts:
        # so sánh với base_dt biểu diễn dd/MM và HH:MM
        base_dm = base_dt.strftime("%d/%m")
        base_hm = base_dt.strftime("%H:%M")
        # nếu có timestamp không giống base -> in log
        different = [t for t in orig_ts if not (t.startswith(base_dm) and t.endswith(base_hm))]
        if different:
            pbar.write(f"[NOTE] Page trả timestamp khác mốc yêu cầu (vd): {different[:3]}. Mình sẽ ghi đè thời gian sang {base_dt.strftime('%d/%m/%Y %H:%M')}")

    # detect time columns
    time_cols = detect_time_columns(df)
    if time_cols:
        pbar.write(f"[INFO] Cột thời gian phát hiện: {time_cols}")
    else:
        pbar.write("[INFO] Không tự động phát hiện được cột thời gian; sẽ không ghi đè ô thời gian.")

    # CHỈ ghi đè trong các cột time_cols: thay pattern RE_DDMM_HHMM bằng base_dt string
    target = base_dt.strftime("%d/%m/%Y %H:%M")
    for c in time_cols:
        # thực hiện replace regex trên series (an toàn với NaN)
        df[c] = df[c].astype(str).replace(RE_DDMM_HHMM, target, regex=True).map(lambda v: normalize_ws(v))

    # Cắt 'Đồng bộ lúc...' CHỈ trong cột 'Tên hồ' (nếu tồn tại)
    # cột có thể khác tên, nhưng theo bạn là "Tên hồ"
    if "Tên hồ" in df.columns:
        df["Tên hồ"] = cut_sync_tail_in_tenho_series(df["Tên hồ"])

    # Xóa header vùng và note rows
    df = drop_region_header_rows(df)
    df = drop_note_rows(df)
    return df

# HTTP & parse (giữ nguyên, có logging)
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
    # clean + CHỈ ghi đè cột thời gian về dt
    df_all = clean_and_override_time_columns(df_all, dt, pbar)
    # khử trùng trong 1 mốc
    df_all = df_all.drop_duplicates()
    return df_all

def generate_datetimes(start_date: datetime, end_date: datetime, hours: Iterable[int]) -> List[datetime]:
    cur = datetime(start_date.year, start_date.month, start_date.day)
    end = datetime(end_date.year, end_date.month, end_date.day)
    out = []
    while cur <= end:
        for h in hours:
            out.append(cur.replace(hour=h, minute=0, second=0, microsecond=0))
        cur += timedelta(days=1)
    return out

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
        print(f"[INFO] Đã xóa file cũ: {out_path.resolve()}")

    batch = []
    seen_td = set()

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

            if len(batch) >= flush_every:
                combined = pd.concat(batch, ignore_index=True, sort=False)
                # không drop_duplicates ở cấp batch (giữ những bản ghi giống nhau giữa mốc khác nhau)
                if out_path.exists():
                    combined.to_csv(out_path, mode="a", header=False, index=False, encoding="utf-8-sig")
                else:
                    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
                pbar.write(f"[FLUSH] Ghi {len(combined)} dòng. Tiến độ {i}/{len(dts)}.")
                batch = []

            if polite_delay > 0:
                time.sleep(polite_delay)

            pbar.update(1)

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
        # end_date="01/01/2022",
        # hours=[0],
        hours=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        hc="3-27-44",
        out_csv="data_thuydien_new.csv",
        flush_every=120,
        polite_delay=0,
        overwrite=True
    )
