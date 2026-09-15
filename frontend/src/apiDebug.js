const endpoints = [
  "/api/crops",
  "/api/v1/crops",
  "/api/v1/market-prices",
  "/api/v1/market-prices/weeks",
  "/api/v1/market-prices/weekly",
  "/api/v1/markets",
  "/api/v1/markets/weeks",
  "/api/v1/market-data",
];

async function testEndpoint(endpoint) {
  try {
    const response = await fetch(endpoint);
    const text = await response.text();

    console.log("================================");
    console.log("ENDPOINT:", endpoint);
    console.log("STATUS:", response.status);
    console.log("RESPONSE:", text);

    return { endpoint, status: response.status, response: text };
  } catch (error) {
    console.error("ENDPOINT:", endpoint);
    console.error("ERROR:", error);
    return { endpoint, error: error.message };
  }
}

export async function debugMarketAPI() {
  for (const endpoint of endpoints) {
    await testEndpoint(endpoint);
  }
}