-- Database Schema for Agricultural Market Intelligence

CREATE TABLE IF NOT EXISTS canonical_market_prices (
    id SERIAL PRIMARY KEY,
    commodity VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    district VARCHAR(100) NOT NULL,
    market VARCHAR(100) NOT NULL,
    year INT NOT NULL,
    month VARCHAR(20) NOT NULL,
    week INT,
    modal_price_per_quintal NUMERIC(10, 2),
    unit VARCHAR(50) DEFAULT 'Rs./Quintal',
    source_file VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_market_weekly_observation UNIQUE (district, market, year, month, week)
);

CREATE INDEX IF NOT EXISTS idx_market_time ON canonical_market_prices (market, year, month);