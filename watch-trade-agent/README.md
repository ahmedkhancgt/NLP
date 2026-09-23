# Watch Trade Agent (n8n + PostgreSQL + Claude)

An AI assistant for trading new and pre-owned luxury watches internationally, built as **13 n8n workflows**
and a **PostgreSQL database**.

It finds market prices, calculates fair prices, finds underpriced deals, screens authenticity, writes
multilingual listings, answers customer emails, sends matching alerts and campaigns, wins back inactive
customers, tracks shipments and sends a weekly market report.

> **Not included yet (by design):** WhatsApp, Twilio SMS and the voice agent. They will be added later on top
> of the same database. The `messages_log.channel` column is ready for them.

> **Always human:** payment, receipts, negotiation, discounts and large deals. The AI hands these to you and
> never agrees to a price or a payment.

---

## 1. How it fits together

```
                 ┌──────────────────── PostgreSQL ────────────────────┐
                 │ market_prices · deals · watches · listings        │
                 │ customers · interests · inquiries · shipments     │
                 │ settings (your business rules)                    │
                 └───────▲───────────────▲───────────────▲───────────┘
                         │               │               │
   MARKET                │   STOCK       │   CUSTOMERS   │
   01 Currency rates ────┤   05 Intake + authenticity ───┤   08 Reply to every email
   02 eBay prices   ─────┤   06 Multilingual listings ───┤   07 Instant match alerts
   03 Add price (form) ──┤   12 Sale + shipping + DHL ───┤   09 Weekly interest campaign
   04 Deal finder ───────┤                               │   10 Win back inactive
   11 Weekly report ─────┘                               │   13 Add customer (form)
                                                         │
                                 Claude API (writing, reading photos, analysis)
```

| # | Workflow | Runs | What it does |
|---|----------|------|--------------|
| 01 | Currency rates | daily | Official ECB exchange rates (free), plus Gulf currencies |
| 02 | Collect eBay prices | every 6 h | eBay official API in 5 countries; removes junk listings |
| 03 | Add a price by hand | form | Auction results, dealers, Chrono24, forums, brand retail prices |
| 04 | Deal finder | hourly | Listings X% below fair price → email to you (flags "too cheap to be true") |
| 05 | Intake + authenticity | form | Adds a watch, suggests a price, Claude reads serial/papers photos and lists warning signs; second form for the watchmaker's verdict |
| 06 | Multilingual listings | every 15 min | Claude writes listings in EN, AR, FR, DE, ZH (configurable) |
| 07 | Instant match alerts | every 15 min | Customers whose interests match a new watch get an email in their language |
| 08 | Reply to every inquiry | new email | Personal reply using your real stock, in the customer's language; payment, discounts and big deals → you |
| 09 | Weekly interest campaign | Mondays | Personal email with up to 5 new matching watches |
| 10 | Win back inactive | monthly | "We miss you" email to customers quiet for 180+ days |
| 11 | Weekly market report | Mondays | Competitors, cheapest countries, above-retail models, seasonality, stock vs market, with an AI summary |
| 12 | Sale, shipment, tracking | form + every 6 h | Records the sale (after human payment check), emails tracking, updates the customer on DHL status changes |
| 13 | Add a customer | form | Build the customer database (past buyers, leads, interests, marketing consent) |

---

## 2. How the fair price works (simple version)

1. Every price seen in the last 90 days is converted to **USD** (`fx_rates`).
2. Each price is converted to a **"new condition" price**: a watch in *good* condition is worth 78% of
   new, so we divide by 0.78 (`condition_factors`).
3. Outliers (less than half or more than double the middle price) are removed.
4. A **weighted average** is taken: auction results count 1.0, eBay asking prices 0.6, forums 0.4... (`source_weights`).
5. For one specific watch: `fair price = new price × condition factor × (−8% if no box/papers)`.

Try it in SQL:

```sql
SELECT * FROM v_fair_prices;                              -- fair "new" price per reference
SELECT fair_price('126610LN', 'excellent', true);         -- one watch
SELECT * FROM v_country_prices WHERE reference = '126610LN' ORDER BY avg_new_equiv_usd;  -- cheapest country
```

All numbers (deal threshold, weights, condition factors) are in tables you can edit. No code changes needed.

---

## 3. Setup, step by step

### Step 1: Database
Create a PostgreSQL database (for example free **Supabase** or **Neon**, or your own server) and run:

```bash
psql "postgresql://USER:PASSWORD@HOST:5432/DBNAME" -f database/schema.sql
psql "postgresql://USER:PASSWORD@HOST:5432/DBNAME" -f database/sample_data.sql   # optional test data
```

Then put in your own details:

```sql
UPDATE settings SET value = 'Your Company Name'   WHERE key = 'business_name';
UPDATE settings SET value = 'you@yourcompany.com' WHERE key = 'owner_email';
UPDATE settings SET value = 'sales@yourcompany.com' WHERE key = 'from_email';
```

### Step 2: n8n
Use **n8n Cloud** or self-host n8n (Docker). The workflows were tested with n8n 2.40. Then in n8n: **Workflows → Import from file**, and import
every file in `workflows/`.

### Step 3: Credentials (n8n → Credentials → Add)

| Credential | Type in n8n | Used by | Where to get it |
|------------|-------------|---------|-----------------|
| Database | **Postgres** | all | Your database host, name, user, password |
| Email sending | **SMTP** | 04–12 | Your email provider (Google Workspace, Microsoft 365, Brevo, SendGrid...) |
| Email inbox | **IMAP** | 08 | Same mailbox customers write to |
| Claude | **Header Auth**, name `x-api-key`, value = your key | 05, 06, 08–11 | console.anthropic.com → API keys |
| eBay | **OAuth2 API**, grant type *Client Credentials*, token URL `https://api.ebay.com/identity/v1/oauth2/token`, scope `https://api.ebay.com/oauth/api_scope` | 02 | developer.ebay.com (free) |
| DHL | **Header Auth**, name `DHL-API-Key` | 12 | developer.dhl.com, "Shipment Tracking - Unified" (free) |

Open each workflow and pick the right credential on every node that shows a red warning.

### Step 4: Test, then switch on
1. Open a workflow and click **Execute workflow** to test it.
2. For forms, open the form node and use the **Test URL**.
3. When everything works, switch the workflow to **Active**. Forms then use the **Production URL**.

Suggested order: 01 → 13 → 03 → 02 → 05 → 06 → 07 → 04 → 08 → 12 → 09 → 10 → 11.

---

## 4. Everyday use

| You want to... | Do this |
|----------------|---------|
| Add a watch to stock | Fill the **intake form** (workflow 05) with photos |
| Confirm a watch is genuine | Watchmaker fills the **verdict form** (workflow 05) |
| Add a price you saw at an auction or dealer | **Price form** (workflow 03) |
| Add a customer | **Customer form** (workflow 13) |
| Record a sale (after payment is confirmed) | **Sale form** (workflow 12) |
| Search more models | `INSERT INTO tracked_references (brand, model, reference) VALUES ('Tudor', 'Black Bay 58', '79030N');` |
| Follow a competitor | `INSERT INTO competitors (seller) VALUES ('their_ebay_or_dealer_name');` |
| Change the deal threshold | `UPDATE settings SET value = '15' WHERE key = 'deal_threshold_pct';` |
| Review AI replies before sending | `UPDATE settings SET value = 'false' WHERE key = 'auto_send_replies';` |

### Importing past buyers from a spreadsheet
Save the sheet as CSV with the columns `name,email,phone,country,language,marketing_opt_in`, then:

```sql
\copy customers (name, email, phone, country, language, marketing_opt_in) FROM 'buyers.csv' CSV HEADER
```

Only mark `marketing_opt_in = true` for customers who agreed to receive marketing emails.

---

## 5. Price sources and rules

| Source | How | Notes |
|--------|-----|-------|
| eBay | Official Browse API (workflow 02) | Asking prices. Sold prices need eBay's Marketplace Insights approval |
| Auctions (Phillips, Christie's, Sotheby's, Bonhams, Antiquorum, Monaco Legend) | Form (03) | Real sold prices, the most trusted source |
| Dealers, Chrono24, forums | Form (03) | Chrono24 has no public API and its terms forbid scraping |
| Brand retail (Rolex.com, Omega...) | Form (03), source `brand_retail` | Used for the "above retail" report, not for the fair price |
| WatchCharts (paid) | Add an HTTP Request node to workflow 02 | Save with source `watchcharts` |
| Exchange rates | ECB via Frankfurter (01) | Free, no key |

---

## 6. Important limits

- **Authenticity:** Claude only does a first screening of photos and papers. It never certifies a watch.
  Only a watchmaker's inspection sets `watchmaker_verified`. Check stolen-watch registers (for example The Watch
  Register) before buying expensive pieces.
- **Marketing consent:** campaigns and alerts go only to customers with `marketing_opt_in = true`. "STOP"
  unsubscribes them automatically.
- **International shipping:** declare the full value for customs. Straps made of alligator or other exotic
  leather need **CITES** paperwork, so ship them on a different strap when in doubt.
- **AI model:** the workflows use `claude-opus-5`. To change it, edit the `model:` line in the
  "Build ... request" Code nodes.

## 7. Approximate running costs (monthly)

| Item | Estimate |
|------|----------|
| n8n Cloud or a small self-hosted server | $10–60 |
| PostgreSQL (Supabase/Neon) | $0–25 |
| Claude API (listings, replies, campaigns, reports) | $50–400, depending on email volume |
| Email sending | $0–30 |
| eBay, DHL, ECB APIs | free |
