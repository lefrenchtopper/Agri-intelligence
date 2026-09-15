import React from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ForecastResponse, MarketPrice } from '../types/api';

interface ChartProps {
  prices: MarketPrice[];
  forecast: ForecastResponse | null;
}

export const MarketPriceChart: React.FC<ChartProps> = ({ prices, forecast }) => {
  const formattedData = prices
    .slice()
    .sort(
      (a, b) =>
        new Date(a.price_date || a.week_start || 0).getTime() -
        new Date(b.price_date || b.week_start || 0).getTime(),
    )
    .map((price) => {
      const rawDate = price.price_date || price.week_start;
      const displayDate = rawDate
        ? new Date(rawDate).toLocaleDateString('en-IN', {
            month: 'short',
            day: 'numeric',
            year: '2-digit',
          })
        : price.week && price.year
          ? `W${price.week} ${price.year}`
          : 'N/A';

      return {
        date: displayDate,
        Price:
          price.modal_price_per_quintal ??
          price.modal_price ??
          price.average_price ??
          price.average_price_per_quintal ??
          0,
        Forecast: undefined,
      };
    });

  if (forecast && formattedData.length > 0) {
    formattedData.push({
      date: 'Forecast',
      Price: undefined,
      Forecast: forecast.forecast.predicted_price,
    });
  }

  return (
    <section className="price-chart-section" aria-label="Market price trend">
      {forecast?.spike_alert && (
        <div className="chart-alert">
          <strong>Spike Alert:</strong> Predicted price increase of{' '}
          {(forecast.forecast.predicted_pct_change * 100).toFixed(1)}% expected in{' '}
          {forecast.district}.
        </div>
      )}

      <div className="price-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={formattedData} margin={{ top: 10, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#dfe4dc" />
            <XAxis dataKey="date" tick={{ fill: '#697269', fontSize: 11 }} />
            <YAxis
              domain={['auto', 'auto']}
              tick={{ fill: '#697269', fontSize: 11 }}
              label={{ value: 'INR / Quintal', angle: -90, position: 'insideLeft', fill: '#697269' }}
            />
            <Tooltip formatter={(value: number) => [`₹${value.toFixed(2)}`, 'Price']} />
            <Legend />
            <Line type="monotone" dataKey="Price" stroke="#16a34a" strokeWidth={2} dot={false} />
            {forecast && (
              <Line
                type="monotone"
                dataKey="Forecast"
                stroke="#d97706"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ r: 4 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
};
