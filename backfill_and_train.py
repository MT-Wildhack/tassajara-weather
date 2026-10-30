def _pull_point_chunked(lat: float, lon: float, elev: float) -> pd.DataFrame:
        frames = []
        cur_start = start_date
        while cur_start <= end_date:
            cur_end = min(cur_start + timedelta(days=180), end_date)
            params = {
                "latitude": lat,
                "longitude": lon,
                "elevation": elev,
                "start_date": cur_start.isoformat(),
                "end_date": cur_end.isoformat(),
                "hourly": (
                    "temperature_2m,relative_humidity_2m,dew_point_2m,"
                    "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
                    "precipitation_probability,precipitation,cloud_cover"
                ),
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            }
            h = None
            for attempt in range(3):
                try:
                    resp = requests.get(
                        "https://historical-forecast-api.open-meteo.com/v1/forecast",
                        params=params,
                        timeout=35,
                    )
                    if resp.status_code == 429:
                        time.sleep(2.5 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    h = resp.json().get("hourly")
                    break
                except Exception:
                    try:
                        fallback_params = dict(params)
                        fallback_params["hourly"] = (
                            "temperature_2m,relative_humidity_2m,dew_point_2m,"
                            "wind_speed_10m,wind_gusts_10m,wind_direction_10m,"
                            "precipitation,cloud_cover"
                        )
                        resp = requests.get(
                            "https://archive-api.open-meteo.com/v1/archive",
                            params=fallback_params,
                            timeout=35,
                        )
                        if resp.status_code == 429:
                            time.sleep(2.5 * (attempt + 1))
                            continue
                        resp.raise_for_status()
                        h = resp.json().get("hourly")
                        break
                    except Exception as exc:
                        if attempt == 2:
                            logger.warning(f"Open-Meteo chunk {cur_start}..{cur_end} failed: {exc}")
                        time.sleep(1.5 * (attempt + 1))

            if h:
                df_chunk = pd.DataFrame(h)
                if "precipitation_probability" not in df_chunk.columns:
                    p_mm = pd.to_numeric(df_chunk.get("precipitation", 0.0), errors="coerce").fillna(0.0)
                    cc = pd.to_numeric(df_chunk.get("cloud_cover", 20.0), errors="coerce").fillna(20.0)
                    rh = pd.to_numeric(df_chunk.get("relative_humidity_2m", 50.0), errors="coerce").fillna(50.0)
                    df_chunk["precipitation_probability"] = np.clip(
                        np.where(
                            p_mm >= 0.25,
                            65.0 + 15.0 * np.log1p(p_mm),
                            (cc * 0.25) * np.maximum(0.0, (rh - 60.0) / 40.0),
                        ),
                        0.0,
                        100.0,
                    ).round(0)
                frames.append(df_chunk)
            cur_start = cur_end + timedelta(days=1)
            time.sleep(0.35)

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"])
        dt_series = pd.to_datetime(df["time"], utc=True)
        df["valid_time_utc"] = dt_series.map(lambda x: x.isoformat())
        df = df[dt_series <= now_utc].reset_index(drop=True)
        return df