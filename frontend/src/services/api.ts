import { Crop, ForecastResponse, MarketPrice } from '../types/api';

const BASE_URL = 'http://127.0.0.1:8000';

export async function fetchCrops(): Promise<Crop[]> {
  const response = await fetch(`${BASE_URL}/api/crops`);
  if (!response.ok) throw new Error('Failed to fetch crops');
  return response.json();
}

export async function fetchDistrictForecast(district: string): Promise<ForecastResponse> {
  const formattedDistrict = district.charAt(0).toUpperCase() + district.slice(1).toLowerCase();
  const response = await fetch(`${BASE_URL}/api/v1/forecasts/${formattedDistrict}`);
  if (!response.ok) throw new Error(`Failed to fetch forecast for ${district}`);
  return response.json();
}

export async function fetchWeeklyPrices(cropId?: string, district?: string): Promise<MarketPrice[]> {
  const params = new URLSearchParams();
  if (cropId) params.append('crop_id', cropId);
  if (district) params.append('district', district);

  const response = await fetch(`${BASE_URL}/api/market-prices/weekly?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch market prices');
  return response.json();
}

export interface MarketTableFilters {
  crop_id: string;
  district: string;
  year: number | string;
  month: number | string;
  week: number | string;
}

export async function fetchFilteredMarketTable(
  filters: MarketTableFilters,
): Promise<MarketPrice[]> {
  const params = new URLSearchParams({
    crop_id: filters.crop_id,
    district: filters.district,
    year: String(filters.year),
    month: String(filters.month),
    week: String(filters.week),
  });

  const response = await fetch(`${BASE_URL}/api/market-prices/weekly?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch market prices');
  return response.json();
}