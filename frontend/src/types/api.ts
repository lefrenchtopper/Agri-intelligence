export interface Crop {
  id: string;
  name: string;
  tamil_name: string;
  category: string;
  growing_days_min: number;
  growing_days_max: number;
}

export interface ForecastData {
  predicted_price: number;
  lower_bound: number;
  upper_bound: number;
  predicted_pct_change: number;
}

export interface ForecastResponse {
  status: string;
  district: string;
  current_price: number;
  forecast: ForecastData;
  regime: string;
  spike_alert: boolean;
}

export interface MarketPrice {
  id: string;
  crop_id: string;
  district: string;
  market_name: string;
  modal_price_per_quintal?: number;
  modal_price?: number;
  average_price?: number;
  average_price_per_quintal?: number;
  min_price?: number;
  max_price?: number;
  price_date?: string;
  week_start?: string;
  year: number;
  month: number;
  week: number;
}