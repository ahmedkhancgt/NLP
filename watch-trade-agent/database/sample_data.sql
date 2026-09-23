-- =====================================================================
-- EXAMPLE DATA ONLY - made-up prices and people, for testing the workflows.
-- Delete it before going live:
--     TRUNCATE market_prices, deals, customer_interests, messages_log,
--              inquiries, purchases, shipments, listings,
--              authenticity_checks, watches, customers RESTART IDENTITY CASCADE;
-- =====================================================================

-- References the robot searches for
INSERT INTO tracked_references (brand, model, reference) VALUES
  ('Rolex',          'Submariner Date',  '126610LN'),
  ('Rolex',          'GMT-Master II',    '126710BLRO'),
  ('Omega',          'Speedmaster Moonwatch', '310.30.42.50.01.001'),
  ('Audemars Piguet','Royal Oak',        '15510ST'),
  ('Patek Philippe', 'Aquanaut',         '5167A');

-- Competitors to follow (use the seller name exactly as it appears in market_prices)
INSERT INTO competitors (seller, website) VALUES
  ('example_dealer_uk', 'https://example.com'),
  ('example_dealer_us', 'https://example.org');

-- A few exchange rates (the fx workflow keeps these up to date)
INSERT INTO fx_rates (currency, per_usd) VALUES
  ('EUR', 0.92), ('GBP', 0.79), ('AED', 3.6725), ('JPY', 148), ('CHF', 0.88), ('HKD', 7.8)
ON CONFLICT (currency) DO NOTHING;

-- Example market prices for the Submariner 126610LN (made-up numbers)
INSERT INTO market_prices (source, external_id, seller, brand, reference, title, condition, box_papers, price, currency, price_usd, country, url) VALUES
  ('auction',  'lot-101', 'Example Auction', 'Rolex', '126610LN', 'Rolex Submariner 126610LN', 'excellent', true, 12500, 'USD', 12500, 'US', NULL),
  ('auction',  'lot-102', 'Example Auction', 'Rolex', '126610LN', 'Rolex Submariner 126610LN', 'very_good', true, 11900, 'USD', 11900, 'CH', NULL),
  ('dealer',   'd-201',   'example_dealer_uk', 'Rolex', '126610LN', 'Submariner Date 2022 full set', 'excellent', true, 10900, 'GBP', ROUND(10900/0.79), 'GB', NULL),
  ('dealer',   'd-202',   'example_dealer_us', 'Rolex', '126610LN', 'Submariner Date 2021', 'very_good', true, 13200, 'USD', 13200, 'US', NULL),
  ('chrono24', 'c-301',   'some_seller',       'Rolex', '126610LN', 'Rolex Submariner Date', 'unworn', true, 13900, 'EUR', ROUND(13900/0.92), 'DE', NULL),
  ('ebay',     'e-401',   'seller_a',          'Rolex', '126610LN', 'Rolex Submariner 126610LN', 'good', false, 10400, 'USD', 10400, 'US', 'https://www.ebay.com/'),
  ('ebay',     'e-402',   'seller_b',          'Rolex', '126610LN', 'Rolex Submariner 126610LN 2023', 'excellent', true, 9800, 'USD', 9800, 'US', 'https://www.ebay.com/'),
  ('brand_retail', 'rolex-us-126610LN', 'Rolex', 'Rolex', '126610LN', 'Retail price', 'new', true, 10250, 'USD', 10250, 'US', 'https://www.rolex.com/');

-- Example customers (only opted-in customers receive marketing)
INSERT INTO customers (name, email, country, language, marketing_opt_in, last_contact_at, last_purchase_at) VALUES
  ('Sara Example',  'sara@example.com',  'AE', 'ar', true,  now() - interval '10 days',  now() - interval '400 days'),
  ('Tom Example',   'tom@example.com',   'GB', 'en', true,  now() - interval '300 days', now() - interval '300 days'),
  ('Lena Example',  'lena@example.com',  'DE', 'de', false, now() - interval '5 days',   NULL);

INSERT INTO customer_interests (customer_id, brand, reference, max_budget_usd, source) VALUES
  (1, 'Rolex', '126610LN', 15000, 'manual'),
  (2, 'Rolex', NULL,       20000, 'manual'),
  (3, 'Omega', NULL,       NULL,  'manual');
