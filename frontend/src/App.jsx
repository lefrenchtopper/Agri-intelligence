import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const MONTHS = [
  { value: 1, label: "January" },
  { value: 2, label: "February" },
  { value: 3, label: "March" },
  { value: 4, label: "April" },
  { value: 5, label: "May" },
  { value: 6, label: "June" },
  { value: 7, label: "July" },
  { value: 8, label: "August" },
  { value: 9, label: "September" },
  { value: 10, label: "October" },
  { value: 11, label: "November" },
  { value: 12, label: "December" },
];

function App() {
  const [crops, setCrops] = useState([]);
  const [prices, setPrices] = useState([]);

  const [selectedCrop, setSelectedCrop] = useState("");
  const [selectedDistrict, setSelectedDistrict] = useState("Coimbatore");

  const [availableWeeks, setAvailableWeeks] = useState([]);

  const [selectedYear, setSelectedYear] = useState("");
  const [selectedMonth, setSelectedMonth] = useState("");
  const [selectedWeek, setSelectedWeek] = useState("");

  const [loadingCrops, setLoadingCrops] = useState(true);
  const [loadingWeeks, setLoadingWeeks] = useState(false);
  const [loadingPrices, setLoadingPrices] = useState(false);

  const [error, setError] = useState("");

  // ---------------------------------------------------------
  // LOAD CROPS
  // ---------------------------------------------------------

  useEffect(() => {
    async function loadCrops() {
      try {
        setLoadingCrops(true);
        setError("");

        const response = await fetch(`${API_BASE}/api/crops`);

        if (!response.ok) {
          throw new Error("Failed to load crops");
        }

        const data = await response.json();

        setCrops(data);

        // Default to Onion
        const onion = data.find(
          (crop) => crop.name.toLowerCase() === "onion"
        );

        if (onion) {
          setSelectedCrop(onion.id);
        } else if (data.length > 0) {
          setSelectedCrop(data[0].id);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingCrops(false);
      }
    }

    loadCrops();
  }, []);

  // ---------------------------------------------------------
  // LOAD AVAILABLE WEEKS
  // ---------------------------------------------------------

  useEffect(() => {
    if (!selectedCrop) return;

    async function loadAvailableWeeks() {
      try {
        setLoadingWeeks(true);
        setError("");

        const params = new URLSearchParams();

        params.set("crop_id", selectedCrop);

        if (selectedDistrict) {
          params.set("district", selectedDistrict);
        }

        const response = await fetch(
          `${API_BASE}/api/market-prices/weeks?${params.toString()}`
        );

        if (!response.ok) {
          throw new Error("Failed to load available weeks");
        }

        const data = await response.json();

        setAvailableWeeks(data);

        // No data
        if (data.length === 0) {
          setSelectedYear("");
          setSelectedMonth("");
          setSelectedWeek("");
          return;
        }

        // Default to first available date
        const first = data[0];

        setSelectedYear(first.year);
        setSelectedMonth(first.month);
        setSelectedWeek(first.week);
      } catch (err) {
        setError(err.message);
        setAvailableWeeks([]);

        setSelectedYear("");
        setSelectedMonth("");
        setSelectedWeek("");
      } finally {
        setLoadingWeeks(false);
      }
    }

    loadAvailableWeeks();
  }, [selectedCrop, selectedDistrict]);

  // ---------------------------------------------------------
  // AVAILABLE YEARS
  // ---------------------------------------------------------

  const availableYears = useMemo(() => {
    const years = [...new Set(
      availableWeeks.map((item) => item.year)
    )];

    return years.sort((a, b) => a - b);
  }, [availableWeeks]);

  // ---------------------------------------------------------
  // AVAILABLE MONTHS
  // ---------------------------------------------------------

  const availableMonths = useMemo(() => {
    const months = [
      ...new Set(
        availableWeeks
          .filter((item) => item.year === selectedYear)
          .map((item) => item.month)
      ),
    ];

    return months.sort((a, b) => a - b);
  }, [availableWeeks, selectedYear]);

  // ---------------------------------------------------------
  // AVAILABLE WEEKS FOR SELECTED YEAR + MONTH
  // ---------------------------------------------------------

  const filteredWeeks = useMemo(() => {
    return availableWeeks.filter(
      (item) =>
        item.year === selectedYear &&
        item.month === selectedMonth
    );
  }, [availableWeeks, selectedYear, selectedMonth]);

  // ---------------------------------------------------------
  // HANDLE YEAR CHANGE
  // ---------------------------------------------------------

  function handleYearChange(event) {
    const year = Number(event.target.value);

    setSelectedYear(year);

    const monthsForYear = availableWeeks.filter(
      (item) => item.year === year
    );

    const uniqueMonths = [
      ...new Set(
        monthsForYear.map((item) => item.month)
      ),
    ].sort((a, b) => a - b);

    if (uniqueMonths.length > 0) {
      const month = uniqueMonths[0];

      setSelectedMonth(month);

      const weeksForMonth = monthsForYear.filter(
        (item) => item.month === month
      );

      if (weeksForMonth.length > 0) {
        setSelectedWeek(weeksForMonth[0].week);
      }
    } else {
      setSelectedMonth("");
      setSelectedWeek("");
    }
  }

  // ---------------------------------------------------------
  // HANDLE MONTH CHANGE
  // ---------------------------------------------------------

  function handleMonthChange(event) {
    const month = Number(event.target.value);

    setSelectedMonth(month);

    const weeksForMonth = availableWeeks.filter(
      (item) =>
        item.year === selectedYear &&
        item.month === month
    );

    if (weeksForMonth.length > 0) {
      setSelectedWeek(weeksForMonth[0].week);
    } else {
      setSelectedWeek("");
    }
  }

  // ---------------------------------------------------------
  // HANDLE WEEK CHANGE
  // ---------------------------------------------------------

  function handleWeekChange(event) {
    setSelectedWeek(Number(event.target.value));
  }

  // ---------------------------------------------------------
  // LOAD MARKET PRICES
  // ---------------------------------------------------------

  useEffect(() => {
    if (
      !selectedCrop ||
      !selectedYear ||
      !selectedMonth ||
      !selectedWeek
    ) {
      return;
    }

    async function loadPrices() {
      try {
        setLoadingPrices(true);
        setError("");

        const params = new URLSearchParams();

        params.set("crop_id", selectedCrop);

        if (selectedDistrict) {
          params.set("district", selectedDistrict);
        }

        params.set("year", selectedYear);
        params.set("month", selectedMonth);
        params.set("week", selectedWeek);

        const response = await fetch(
          `${API_BASE}/api/market-prices/weekly?${params.toString()}`
        );

        if (!response.ok) {
          throw new Error("Failed to load market prices");
        }

        const data = await response.json();

        setPrices(data);
      } catch (err) {
        setError(err.message);
        setPrices([]);
      } finally {
        setLoadingPrices(false);
      }
    }

    loadPrices();
  }, [
    selectedCrop,
    selectedDistrict,
    selectedYear,
    selectedMonth,
    selectedWeek,
  ]);

  // ---------------------------------------------------------
  // SELECTED CROP
  // ---------------------------------------------------------

  const selectedCropData = useMemo(() => {
    return crops.find(
      (crop) => crop.id === selectedCrop
    );
  }, [crops, selectedCrop]);

  // ---------------------------------------------------------
  // SELECTED MONTH NAME
  // ---------------------------------------------------------

  const selectedMonthName = useMemo(() => {
    return (
      MONTHS.find(
        (month) => month.value === selectedMonth
      )?.label || ""
    );
  }, [selectedMonth]);

  // ---------------------------------------------------------
  // CURRENT DATE RANGE
  // ---------------------------------------------------------

  const currentDateRange = useMemo(() => {
    const selectedWeekData = filteredWeeks.find(
      (item) => item.week === selectedWeek
    );

    if (!selectedWeekData) {
      return null;
    }

    const startDate = new Date(
      `${selectedWeekData.week_start}T00:00:00`
    );

    const endDate = new Date(
      `${selectedWeekData.week_end}T00:00:00`
    );

    const startDay = startDate.getDate();
    const endDay = endDate.getDate();

    const monthName = startDate.toLocaleDateString(
      "en-IN",
      {
        month: "long",
      }
    );

    return `${startDay}–${endDay} ${monthName} ${startDate.getFullYear()}`;
  }, [filteredWeeks, selectedWeek]);

  // ---------------------------------------------------------
  // MARKET STATISTICS
  // ---------------------------------------------------------

  const statistics = useMemo(() => {
    if (prices.length === 0) {
      return {
        average: 0,
        highest: null,
        lowest: null,
      };
    }

    const average =
      prices.reduce(
        (sum, price) =>
          sum + price.average_price_per_quintal,
        0
      ) / prices.length;

    const highest = prices.reduce((max, price) =>
      price.average_price_per_quintal >
      max.average_price_per_quintal
        ? price
        : max
    );

    const lowest = prices.reduce((min, price) =>
      price.average_price_per_quintal <
      min.average_price_per_quintal
        ? price
        : min
    );

    return {
      average,
      highest,
      lowest,
    };
  }, [prices]);

  // ---------------------------------------------------------
  // FORMAT PRICE
  // ---------------------------------------------------------

  function formatPrice(value) {
    return `₹${value.toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  }

  // ---------------------------------------------------------
  // FORMAT CHANGE
  // ---------------------------------------------------------

  function formatChange(value) {
    if (value === null || value === undefined) {
      return "—";
    }

    const arrow = value >= 0 ? "↑" : "↓";

    return `${arrow} ${Math.abs(value).toFixed(1)}%`;
  }

  // ---------------------------------------------------------
  // CHANGE CSS CLASS
  // ---------------------------------------------------------

  function getChangeClass(value) {
    if (value === null || value === undefined) {
      return "";
    }

    return value >= 0 ? "positive" : "negative";
  }

  // ---------------------------------------------------------
  // RENDER
  // ---------------------------------------------------------

  return (
    <div className="dashboard">

      <header className="header">
        <div>
          <div className="brand">
            AGRI-INTELLIGENCE
          </div>

          <div className="subtitle">
            Agricultural Decision Support System
          </div>
        </div>

        <div className="location">
          Coimbatore, Tamil Nadu
        </div>
      </header>


      <main>

        {/* HERO */}

        <section className="hero-section">

          <div className="section-label">
            MARKET ANALYSIS
          </div>

          <h1>
            Weekly Wholesale Market Prices
          </h1>

          <p className="description">

            {selectedCropData
              ? `${selectedCropData.name} · Coimbatore, Tamil Nadu · ${
                  currentDateRange ||
                  `${selectedMonthName} ${selectedYear}`
                }`
              : "Loading market data..."}

          </p>

        </section>


        {/* FILTERS */}

        <section className="filters">

          {/* COMMODITY */}

          <div className="filter">

            <label>
              Commodity
            </label>

            <select
              value={selectedCrop}
              onChange={(event) =>
                setSelectedCrop(event.target.value)
              }
              disabled={loadingCrops}
            >

              {loadingCrops ? (
                <option>
                  Loading...
                </option>
              ) : (
                crops.map((crop) => (
                  <option
                    key={crop.id}
                    value={crop.id}
                  >
                    {crop.name}
                  </option>
                ))
              )}

            </select>

          </div>


          {/* STATE */}

          <div className="filter">

            <label>
              State / UT
            </label>

            <select defaultValue="Tamil Nadu">
              <option>
                Tamil Nadu
              </option>
            </select>

          </div>


          {/* DISTRICT */}

          <div className="filter">

            <label>
              District
            </label>

            <select
              value={selectedDistrict}
              onChange={(event) =>
                setSelectedDistrict(
                  event.target.value
                )
              }
            >

              <option>
                Coimbatore
              </option>

            </select>

          </div>


          {/* YEAR */}

          <div className="filter">

            <label>
              Year
            </label>

            <select
              value={selectedYear}
              onChange={handleYearChange}
              disabled={
                loadingWeeks ||
                availableYears.length === 0
              }
            >

              {availableYears.length === 0 ? (
                <option value="">
                  —
                </option>
              ) : (
                availableYears.map((year) => (
                  <option
                    key={year}
                    value={year}
                  >
                    {year}
                  </option>
                ))
              )}

            </select>

          </div>


          {/* MONTH */}

          <div className="filter">

            <label>
              Month
            </label>

            <select
              value={selectedMonth}
              onChange={handleMonthChange}
              disabled={
                loadingWeeks ||
                availableMonths.length === 0
              }
            >

              {availableMonths.length === 0 ? (
                <option value="">
                  —
                </option>
              ) : (
                availableMonths.map((month) => {

                  const monthData =
                    MONTHS.find(
                      (item) =>
                        item.value === month
                    );

                  return (
                    <option
                      key={month}
                      value={month}
                    >
                      {monthData?.label}
                    </option>
                  );
                })
              )}

            </select>

          </div>


          {/* WEEK */}

          <div className="filter">

            <label>
              Week
            </label>

            <select
              value={selectedWeek}
              onChange={handleWeekChange}
              disabled={
                loadingWeeks ||
                filteredWeeks.length === 0
              }
            >

              {filteredWeeks.length === 0 ? (
                <option value="">
                  —
                </option>
              ) : (
                filteredWeeks.map((item) => (
                  <option
                    key={`${item.year}-${item.month}-${item.week}`}
                    value={item.week}
                  >
                    Week {item.week}
                  </option>
                ))
              )}

            </select>

          </div>

        </section>


        {/* ERROR */}

        {error && (
          <div className="error">
            {error}
          </div>
        )}


        {/* NO DATA */}

        {!loadingPrices &&
          prices.length === 0 &&
          !error && (
            <div className="no-data">
              No market data available for this selection.
            </div>
          )}


        {/* STATISTICS */}

        <section className="stats">

          <div className="stat-card">

            <span>
              Markets
            </span>

            <strong>
              {loadingPrices
                ? "—"
                : prices.length}
            </strong>

          </div>


          <div className="stat-card">

            <span>
              Average Price
            </span>

            <strong>

              {loadingPrices
                ? "—"
                : prices.length > 0
                  ? formatPrice(
                      statistics.average
                    )
                  : "—"}

            </strong>

            <small>
              per quintal (100 kg)
            </small>

          </div>


          <div className="stat-card">

            <span>
              Highest Price
            </span>

            <strong>

              {loadingPrices ||
              !statistics.highest
                ? "—"
                : formatPrice(
                    statistics.highest
                      .average_price_per_quintal
                  )}

            </strong>

            {statistics.highest && (
              <small>
                {statistics.highest.market_name}
              </small>
            )}

          </div>


          <div className="stat-card">

            <span>
              Lowest Price
            </span>

            <strong>

              {loadingPrices ||
              !statistics.lowest
                ? "—"
                : formatPrice(
                    statistics.lowest
                      .average_price_per_quintal
                  )}

            </strong>

            {statistics.lowest && (
              <small>
                {statistics.lowest.market_name}
              </small>
            )}

          </div>

        </section>


        {/* MARKET TABLE */}

        <section className="market-section">

          <div className="section-label">
            MARKET-WISE DATA
          </div>

          <h2>
            Wholesale Prices for{" "}
            {selectedCropData?.name || "Crop"}
          </h2>

          <p className="date-range">
            {currentDateRange ||
              "No data available"}
          </p>


          {loadingPrices ? (

            <div className="loading">
              Loading market prices...
            </div>

          ) : prices.length > 0 ? (

            <div className="table-wrapper">

              <table>

                <thead>

                  <tr>

                    <th>
                      Market
                    </th>

                    <th>
                      Price
                      <br />
                      <span>
                        {currentDateRange}
                      </span>
                    </th>

                    <th>
                      Previous Week
                    </th>

                    <th>
                      Previous Month
                    </th>

                    <th>
                      Previous Year
                    </th>

                    <th>
                      Change
                      <br />
                      Previous Week
                    </th>

                    <th>
                      Change
                      <br />
                      Previous Month
                    </th>

                    <th>
                      Change
                      <br />
                      Previous Year
                    </th>

                  </tr>

                </thead>


                <tbody>

                  {prices.map((price) => (

                    <tr key={price.id}>

                      <td>
                        {price.market_name}
                      </td>

                      <td>
                        {formatPrice(
                          price.average_price_per_quintal
                        )}
                      </td>

                      <td>
                        {price.previous_week_price !== null
                          ? formatPrice(
                              price.previous_week_price
                            )
                          : "—"}
                      </td>

                      <td>
                        {price.previous_month_price !== null
                          ? formatPrice(
                              price.previous_month_price
                            )
                          : "—"}
                      </td>

                      <td>
                        {price.previous_year_price !== null
                          ? formatPrice(
                              price.previous_year_price
                            )
                          : "—"}
                      </td>

                      <td
                        className={getChangeClass(
                          price.change_over_previous_week_pct
                        )}
                      >
                        {formatChange(
                          price.change_over_previous_week_pct
                        )}
                      </td>

                      <td
                        className={getChangeClass(
                          price.change_over_previous_month_pct
                        )}
                      >
                        {formatChange(
                          price.change_over_previous_month_pct
                        )}
                      </td>

                      <td
                        className={getChangeClass(
                          price.change_over_previous_year_pct
                        )}
                      >
                        {formatChange(
                          price.change_over_previous_year_pct
                        )}
                      </td>

                    </tr>

                  ))}

                </tbody>

              </table>

            </div>

          ) : null}

        </section>

      </main>


      <footer>

        <span>
          Agri-Intelligence
        </span>

        <span>
          Market data · Coimbatore · ₹/quintal (100 kg)
        </span>

      </footer>

    </div>
  );
}

export default App;