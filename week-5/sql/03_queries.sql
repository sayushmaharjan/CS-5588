USE DATABASE WEATHER_TWIN_DB;
USE SCHEMA PUBLIC;

--------------------------------------------------
-- 1) Aggregation / KPI query
--------------------------------------------------
CREATE OR REPLACE VIEW CITY_STATS AS
SELECT
    CITY,
    COUNT(*)                         AS num_records,
    AVG(TAVG)                  AS avg_temp,
    MIN(TMIN)                  AS min_temp,
    MAX(TMAX)                  AS max_temp,
    -- AVG(WIND_SPEED)                  AS avg_wind,
    COUNT_IF(PRCP > 0)       AS rainy_days
FROM WEATHER_FULL
GROUP BY CITY;

--------------------------------------------------
-- 2) Temporal / filtering query (e.g. last 30 days)
--------------------------------------------------
CREATE OR REPLACE VIEW RECENT_CITY_WEATHER AS
SELECT *
FROM WEATHER_FULL
WHERE DATE >= DATEADD(day, -30, CURRENT_DATE());

--------------------------------------------------
-- 3) Join / feature query (join base table with CITY_STATS)
--------------------------------------------------
CREATE OR REPLACE VIEW V_WEATHER_WITH_CITY_STATS AS
SELECT
    w.*,
    s.avg_temp AS city_avg_temp,
    s.avg_wind AS city_avg_wind,
    s.rainy_days
FROM WEATHER_FULL AS w
LEFT JOIN CITY_STATS AS s
    ON w.CITY = s.CITY;