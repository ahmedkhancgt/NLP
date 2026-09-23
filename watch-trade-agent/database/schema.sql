-- =====================================================================
-- Watch Trade Agent - PostgreSQL database
-- Run this file once on an empty database:
--     psql "$DATABASE_URL" -f database/schema.sql
-- Then (optional) load example data:
--     psql "$DATABASE_URL" -f database/sample_data.sql
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. SETTINGS - business rules you can change without touching n8n
-- ---------------------------------------------------------------------
CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    note  TEXT
);

INSERT INTO settings (key, value, note) VALUES
  ('business_name',          'My Watch Trading Co.',  'Used in emails and listings'),
  ('owner_email',            'owner@example.com',     'Where deal alerts, reports and human hand-offs go'),
  ('from_email',             'sales@example.com',     'Sender address for customer emails'),
  ('deal_threshold_pct',     '12',     'A listing this % below fair price is flagged as a deal'),
  ('suspicious_deal_pct',    '40',     'More than this % below fair price = probably fake or scam'),
  ('min_price_samples',      '5',      'Need at least this many recent prices before trusting a fair price'),
  ('no_papers_discount_pct', '8',      'Fair price discount when box and papers are missing'),
  ('large_deal_usd',         '25000',  'Inquiries above this value always go to a human'),
  ('listing_languages',      'en,ar,fr,de,zh', 'Languages for AI-written listings'),
  ('auto_send_replies',      'true',   'true = AI replies are emailed directly; false = drafts go to owner'),
  ('inactive_days',          '180',    'Customers with no contact for this many days get a win-back email'),
  ('campaign_cooldown_days', '7',      'Minimum days between marketing emails to one customer');


-- ---------------------------------------------------------------------
-- 2. MARKET DATA
-- ---------------------------------------------------------------------

-- Which references the robot should search for
CREATE TABLE tracked_references (
    id        SERIAL PRIMARY KEY,
    brand     TEXT NOT NULL,
    model     TEXT,
    reference TEXT NOT NULL UNIQUE,
    active    BOOLEAN NOT NULL DEFAULT TRUE
);

-- Exchange rates: how many units of a currency equal 1 USD
CREATE TABLE fx_rates (
    currency   CHAR(3) PRIMARY KEY,
    per_usd    NUMERIC NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO fx_rates (currency, per_usd) VALUES ('USD', 1);

-- How much we trust each price source (1.0 = fully, 0 = ignore for pricing)
CREATE TABLE source_weights (
    source     TEXT PRIMARY KEY,
    price_type TEXT NOT NULL CHECK (price_type IN ('sold', 'asking', 'market', 'retail')),
    weight     NUMERIC NOT NULL,
    note       TEXT
);
INSERT INTO source_weights (source, price_type, weight, note) VALUES
  ('auction',      'sold',   1.0, 'Phillips, Christie''s, Sotheby''s, Bonhams, Antiquorum... (entered by form)'),
  ('ebay_sold',    'sold',   0.9, 'eBay sold prices (needs Marketplace Insights approval, or form entry)'),
  ('watchcharts',  'market', 0.9, 'WatchCharts API market price'),
  ('ebay',         'asking', 0.6, 'eBay active listings (official Browse API)'),
  ('dealer',       'asking', 0.6, 'Dealer websites, entered by form'),
  ('chrono24',     'asking', 0.6, 'Chrono24 asking prices, entered by form'),
  ('forum',        'asking', 0.4, 'WatchUSeek, r/Watchexchange'),
  ('brand_retail', 'retail', 0.0, 'Official brand retail price - used for comparison, not for fair price');

-- Condition factors: price of a watch in this condition vs. new
CREATE TABLE condition_factors (
    condition   TEXT PRIMARY KEY,
    factor      NUMERIC NOT NULL,
    description TEXT
);
INSERT INTO condition_factors (condition, factor, description) VALUES
  ('new',       1.00, 'Unworn, stickers, full set'),
  ('unworn',    0.97, 'Never worn, no stickers'),
  ('excellent', 0.90, 'Very light signs of wear'),
  ('very_good', 0.85, 'Light wear, no damage'),
  ('good',      0.78, 'Visible wear, fully working'),
  ('fair',      0.65, 'Heavy wear or needs service');

-- Every price the robot (or you) saw. One row per listing per day.
CREATE TABLE market_prices (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL REFERENCES source_weights(source),
    external_id TEXT NOT NULL,                 -- listing id, lot number, URL...
    seller      TEXT,                          -- dealer / eBay seller name (for competitor tracking)
    brand       TEXT,
    reference   TEXT NOT NULL,
    title       TEXT,
    condition   TEXT NOT NULL DEFAULT 'good' REFERENCES condition_factors(condition),
    box_papers  BOOLEAN,
    price       NUMERIC NOT NULL,
    currency    CHAR(3) NOT NULL,
    price_usd   NUMERIC,
    country     CHAR(2),                       -- where the watch is located
    url         TEXT,
    observed_on DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, external_id, observed_on)
);
CREATE INDEX market_prices_ref_idx ON market_prices (reference, observed_on);

-- Competitors whose prices we follow (seller names as they appear in market_prices)
CREATE TABLE competitors (
    seller  TEXT PRIMARY KEY,
    website TEXT,
    note    TEXT
);

-- Underpriced listings found by the deal finder
CREATE TABLE deals (
    id              SERIAL PRIMARY KEY,
    market_price_id BIGINT NOT NULL UNIQUE REFERENCES market_prices(id),
    fair_usd        NUMERIC NOT NULL,
    price_usd       NUMERIC NOT NULL,
    discount_pct    NUMERIC NOT NULL,
    suspicious      BOOLEAN NOT NULL DEFAULT FALSE,
    status          TEXT NOT NULL DEFAULT 'new'
                    CHECK (status IN ('new', 'contacted', 'bought', 'ignored')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ---------------------------------------------------------------------
-- 3. INVENTORY
-- ---------------------------------------------------------------------
CREATE TABLE watches (
    id                  SERIAL PRIMARY KEY,
    brand               TEXT NOT NULL,
    model               TEXT,
    reference           TEXT NOT NULL,
    serial_number       TEXT,
    year                INT,
    condition           TEXT NOT NULL REFERENCES condition_factors(condition),
    box_papers          BOOLEAN NOT NULL DEFAULT FALSE,
    cost_usd            NUMERIC,
    suggested_price_usd NUMERIC,              -- from fair_price() at intake
    ask_usd             NUMERIC,
    location_country    CHAR(2),
    status              TEXT NOT NULL DEFAULT 'in_stock'
                        CHECK (status IN ('in_stock', 'reserved', 'sold', 'shipped')),
    auth_status         TEXT NOT NULL DEFAULT 'pending'
                        CHECK (auth_status IN ('pending', 'ai_checked', 'watchmaker_verified', 'rejected')),
    listings_done       BOOLEAN NOT NULL DEFAULT FALSE,
    alerts_sent         BOOLEAN NOT NULL DEFAULT FALSE,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    sold_at             TIMESTAMPTZ
);

-- AI help for authenticity. The AI NEVER marks a watch as genuine:
-- a watchmaker must set watches.auth_status = 'watchmaker_verified'.
CREATE TABLE authenticity_checks (
    id                SERIAL PRIMARY KEY,
    watch_id          INT NOT NULL REFERENCES watches(id),
    serial_read       TEXT,
    reference_read    TEXT,
    serial_matches    BOOLEAN,
    reference_matches BOOLEAN,
    papers_consistent BOOLEAN,
    risk_flags        TEXT,
    ai_confidence     TEXT,
    ai_notes          TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- AI-written listings, one per language
CREATE TABLE listings (
    id          SERIAL PRIMARY KEY,
    watch_id    INT NOT NULL REFERENCES watches(id),
    language    TEXT NOT NULL,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (watch_id, language)
);


-- ---------------------------------------------------------------------
-- 4. CUSTOMERS
-- ---------------------------------------------------------------------
CREATE TABLE customers (
    id               SERIAL PRIMARY KEY,
    name             TEXT,
    email            TEXT NOT NULL UNIQUE,
    phone            TEXT,
    country          CHAR(2),
    language         TEXT NOT NULL DEFAULT 'en',
    marketing_opt_in BOOLEAN NOT NULL DEFAULT FALSE,   -- only TRUE customers get campaigns
    notes            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_contact_at  TIMESTAMPTZ,
    last_purchase_at TIMESTAMPTZ
);

-- What each customer wants. reference / max_budget_usd can be empty (= any).
CREATE TABLE customer_interests (
    id             SERIAL PRIMARY KEY,
    customer_id    INT NOT NULL REFERENCES customers(id),
    brand          TEXT NOT NULL,
    reference      TEXT,
    max_budget_usd NUMERIC,
    source         TEXT NOT NULL DEFAULT 'manual',  -- manual / inquiry / purchase
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX customer_interests_unique
    ON customer_interests (customer_id, lower(brand), lower(coalesce(reference, '')));

CREATE TABLE purchases (
    id             SERIAL PRIMARY KEY,
    customer_id    INT NOT NULL REFERENCES customers(id),
    watch_id       INT NOT NULL REFERENCES watches(id),
    sale_price_usd NUMERIC NOT NULL,      -- entered by a human after payment is confirmed
    sold_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE inquiries (
    id           SERIAL PRIMARY KEY,
    customer_id  INT REFERENCES customers(id),
    from_email   TEXT NOT NULL,
    subject      TEXT,
    body         TEXT,
    ai_reply     TEXT,
    needs_human  BOOLEAN NOT NULL DEFAULT FALSE,
    human_reason TEXT,
    status       TEXT NOT NULL,          -- auto_sent / sent_to_owner / unsubscribed
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every automatic message we send (prevents double-sending)
CREATE TABLE messages_log (
    id          SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    channel     TEXT NOT NULL DEFAULT 'email',
    campaign    TEXT NOT NULL,           -- match_alert / interest_campaign / reengagement / inquiry_reply / shipping
    watch_id    INT REFERENCES watches(id),
    subject     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX messages_log_customer_idx ON messages_log (customer_id, campaign, created_at);


-- ---------------------------------------------------------------------
-- 5. SHIPPING
-- ---------------------------------------------------------------------
CREATE TABLE shipments (
    id                  SERIAL PRIMARY KEY,
    watch_id            INT NOT NULL REFERENCES watches(id),
    customer_id         INT NOT NULL REFERENCES customers(id),
    carrier             TEXT NOT NULL DEFAULT 'DHL',
    tracking_number     TEXT NOT NULL,
    destination_country CHAR(2),
    status              TEXT NOT NULL DEFAULT 'created',
    status_detail       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    delivered_at        TIMESTAMPTZ
);


-- =====================================================================
-- 6. PRICING LOGIC (views and functions)
-- =====================================================================

-- Small helper: read a number from settings
CREATE FUNCTION setting_num(p_key TEXT) RETURNS NUMERIC
LANGUAGE sql STABLE AS $$
    SELECT value::NUMERIC FROM settings WHERE key = p_key
$$;

-- Fair price per reference, expressed as a "new condition" price in USD.
-- Steps:
--   1. take the last 90 days of prices
--   2. convert each price to "new-equivalent" (divide by condition factor)
--   3. drop outliers (less than half or more than double the median)
--   4. weighted average using source_weights
CREATE VIEW v_fair_prices AS
WITH recent AS (
    SELECT mp.reference,
           mp.brand,
           mp.price_usd / cf.factor AS new_equiv_usd,
           sw.weight
    FROM market_prices mp
    JOIN source_weights    sw ON sw.source = mp.source AND sw.weight > 0
    JOIN condition_factors cf ON cf.condition = mp.condition
    WHERE mp.observed_on >= CURRENT_DATE - 90
      AND mp.price_usd IS NOT NULL
),
medians AS (
    SELECT reference,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY new_equiv_usd) AS median_usd
    FROM recent
    GROUP BY reference
)
SELECT r.reference,
       MAX(r.brand)                                              AS brand,
       ROUND(SUM(r.new_equiv_usd * r.weight) / SUM(r.weight), 0) AS fair_new_usd,
       COUNT(*)                                                  AS samples,
       ROUND(MIN(r.new_equiv_usd), 0)                            AS low_new_usd,
       ROUND(MAX(r.new_equiv_usd), 0)                            AS high_new_usd
FROM recent r
JOIN medians m ON m.reference = r.reference
WHERE r.new_equiv_usd BETWEEN m.median_usd * 0.5 AND m.median_usd * 2
GROUP BY r.reference;

-- Fair price for one watch: reference + condition + box/papers
--   SELECT fair_price('126610LN', 'excellent', true);
CREATE FUNCTION fair_price(p_reference TEXT, p_condition TEXT, p_box_papers BOOLEAN DEFAULT FALSE)
RETURNS NUMERIC
LANGUAGE sql STABLE AS $$
    SELECT ROUND(
             f.fair_new_usd
           * cf.factor
           * CASE WHEN p_box_papers THEN 1
                  ELSE 1 - setting_num('no_papers_discount_pct') / 100 END
           , 0)
    FROM v_fair_prices f
    JOIN condition_factors cf ON cf.condition = p_condition
    WHERE f.reference = p_reference
$$;

-- Average asking price per reference and country (last 30 days) - where is it cheapest?
CREATE VIEW v_country_prices AS
SELECT mp.reference,
       mp.country,
       COUNT(*)                                     AS listings,
       ROUND(AVG(mp.price_usd / cf.factor), 0)      AS avg_new_equiv_usd,
       ROUND(MIN(mp.price_usd), 0)                  AS cheapest_usd
FROM market_prices mp
JOIN condition_factors cf ON cf.condition = mp.condition
JOIN source_weights    sw ON sw.source = mp.source
WHERE mp.observed_on >= CURRENT_DATE - 30
  AND mp.price_usd IS NOT NULL
  AND mp.country IS NOT NULL
  AND sw.price_type IN ('asking', 'sold')
GROUP BY mp.reference, mp.country;

-- Official retail vs. market: which models trade above retail?
CREATE VIEW v_retail_vs_market AS
SELECT f.reference,
       f.brand,
       r.country,
       ROUND(r.price_usd, 0)                            AS retail_usd,
       f.fair_new_usd                                   AS market_new_usd,
       ROUND((f.fair_new_usd / r.price_usd - 1) * 100, 1) AS premium_pct
FROM v_fair_prices f
JOIN LATERAL (
    SELECT DISTINCT ON (mp.country) mp.country, mp.price_usd
    FROM market_prices mp
    WHERE mp.source = 'brand_retail' AND mp.reference = f.reference AND mp.price_usd IS NOT NULL
    ORDER BY mp.country, mp.observed_on DESC
) r ON TRUE;

-- Seasonality: average price and number of listings per calendar month (last 2 years)
CREATE VIEW v_seasonality AS
SELECT mp.reference,
       EXTRACT(MONTH FROM mp.observed_on)::INT     AS month,
       COUNT(*)                                     AS listings,
       ROUND(AVG(mp.price_usd / cf.factor), 0)      AS avg_new_equiv_usd
FROM market_prices mp
JOIN condition_factors cf ON cf.condition = mp.condition
JOIN source_weights    sw ON sw.source = mp.source AND sw.price_type <> 'retail'
WHERE mp.observed_on >= CURRENT_DATE - 730
  AND mp.price_usd IS NOT NULL
GROUP BY mp.reference, EXTRACT(MONTH FROM mp.observed_on);

-- Competitor price moves: this week vs. last week, per competitor and reference
CREATE VIEW v_competitor_moves AS
SELECT mp.seller,
       mp.reference,
       ROUND(AVG(mp.price_usd) FILTER (WHERE mp.observed_on >= CURRENT_DATE - 7), 0)  AS this_week_usd,
       ROUND(AVG(mp.price_usd) FILTER (WHERE mp.observed_on <  CURRENT_DATE - 7), 0)  AS last_week_usd,
       COUNT(*) FILTER (WHERE mp.observed_on >= CURRENT_DATE - 7)                     AS listings_now
FROM market_prices mp
JOIN competitors c ON c.seller = mp.seller
WHERE mp.observed_on >= CURRENT_DATE - 14
GROUP BY mp.seller, mp.reference;
