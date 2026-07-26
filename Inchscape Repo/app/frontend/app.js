const skuInput = document.getElementById("sku-input");
const searchBtn = document.getElementById("search-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const cardTemplate = document.getElementById("variant-card-template");

function clearResults() {
  resultsEl.innerHTML = "";
}

function formatNumber(value) {
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  let cssWidth = canvas.clientWidth;
  let cssHeight = canvas.clientHeight;
  
  // If the canvas hasn't been rendered yet, get the container width
  if (cssWidth <= 0) {
    cssWidth = canvas.parentElement?.clientWidth || 320;
  }
  if (cssHeight <= 0) {
    cssHeight = 190; // fallback to HTML height attribute
  }
  
  console.log('resizeCanvas:', { cssWidth, cssHeight, ratio });
  
  canvas.width = Math.floor(cssWidth * ratio);
  canvas.height = Math.floor(cssHeight * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: cssWidth, height: cssHeight };
}

function drawSeries(ctx, points, xPositions, yScale, color, dashed = false, connectFromPoint = null) {
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash(dashed ? [5, 5] : []);
  
  // If we should connect from a previous point, start there
  if (connectFromPoint != null) {
    ctx.moveTo(connectFromPoint.x, connectFromPoint.y);
  }
  
  let started = connectFromPoint != null;
  points.forEach((value, index) => {
    if (value == null) {
      started = false;
      return;
    }
    const x = xPositions[index];
    const y = yScale(value);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawMarkers(ctx, points, xPositions, yScale, color) {
  ctx.fillStyle = color;
  points.forEach((value, index) => {
    if (value == null) {
      return;
    }
    const x = xPositions[index];
    const y = yScale(value);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawLegend(ctx) {
  const legend = [
    { label: "Actual", color: "#1f6f8b", dashed: false },
    { label: "Test forecast", color: "#ef6c00", dashed: false },
    { label: "Future prediction", color: "#7b2cbf", dashed: true },
  ];

  ctx.font = "12px Space Grotesk";
  legend.forEach((item, i) => {
    const x = 14 + i * 130;
    const y = 18;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.setLineDash(item.dashed ? [5, 5] : []);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 18, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#374151";
    ctx.fillText(item.label, x + 24, y + 4);
  });
}

function drawChart(canvas, result) {
  const history = result.forecast_history ?? [];
  const future = result.future_predictions ?? [];
  const { ctx, width, height } = resizeCanvas(canvas);

  console.log('drawChart called:', { history: history.length, future: future.length, width, height });

  ctx.clearRect(0, 0, width, height);

  if (!history.length && !future.length) {
    ctx.fillStyle = "#5a6271";
    ctx.font = "13px Space Grotesk";
    ctx.fillText("No chart data available.", 14, 28);
    return;
  }

  const labels = [...history.map((p) => p.month), ...future.map((p) => p.month)];
  const futureStartIndex = history.length;
  const actual = [...history.map((p) => p.actual_demand), ...future.map(() => null)];
  const testForecast = [...history.map((p) => p.forecast), ...future.map(() => null)];
  const futureForecast = [...history.map(() => null), ...future.map((p) => p.forecast)];

  console.log('Data arrays:', { actual, testForecast, futureForecast, labels });

  const allValues = [...actual, ...testForecast, ...futureForecast].filter((v) => v != null);
  const minValue = Math.min(...allValues, 0);
  const maxValue = Math.max(...allValues, 1);
  const pad = (maxValue - minValue) * 0.1 || 1;
  const yMin = minValue - pad;
  const yMax = maxValue + pad;

  const margin = { left: 50, right: 10, top: 28, bottom: 26 };
  const chartWidth = Math.max(1, width - margin.left - margin.right);
  const chartHeight = Math.max(1, height - margin.top - margin.bottom);
  const xPositions = labels.map((_, idx) => {
    if (labels.length === 1) {
      return margin.left + chartWidth / 2;
    }
    return margin.left + (idx * chartWidth) / (labels.length - 1);
  });
  const yScale = (v) => margin.top + ((yMax - v) / (yMax - yMin || 1)) * chartHeight;

  if (futureStartIndex < labels.length && futureStartIndex > 0) {
    // Draw future window starting from the last test forecast point (present month)
    const startX = xPositions[futureStartIndex - 1];
    ctx.fillStyle = "rgba(123, 44, 191, 0.08)";
    ctx.fillRect(startX, margin.top, width - margin.right - startX, chartHeight);
    ctx.strokeStyle = "rgba(123, 44, 191, 0.45)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(startX, margin.top);
    ctx.lineTo(startX, margin.top + chartHeight);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#6b21a8";
    ctx.font = "11px Space Grotesk";
    ctx.fillText("future", Math.min(startX + 4, width - margin.right - 36), margin.top + 12);
  }

  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + (i * chartHeight) / 4;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
  }

  drawSeries(ctx, actual, xPositions, yScale, "#1f6f8b");
  drawSeries(ctx, testForecast, xPositions, yScale, "#ef6c00");
  
  // Connect future forecast line from the last test forecast value with a dotted line
  const lastTestIdx = futureStartIndex - 1;
  const firstFutureIdx = futureStartIndex;
  if (lastTestIdx >= 0 && futureStartIndex < xPositions.length && testForecast[lastTestIdx] != null && futureForecast[firstFutureIdx] != null) {
    const lastTestX = xPositions[lastTestIdx];
    const lastTestY = yScale(testForecast[lastTestIdx]);
    const firstFutureX = xPositions[firstFutureIdx];
    const firstFutureY = yScale(futureForecast[firstFutureIdx]);
    
    // Draw connecting dotted line (purple to match future line style)
    ctx.strokeStyle = "#7b2cbf";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(lastTestX, lastTestY);
    ctx.lineTo(firstFutureX, firstFutureY);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  
  drawSeries(ctx, futureForecast, xPositions, yScale, "#7b2cbf", true);
  drawMarkers(ctx, futureForecast, xPositions, yScale, "#7b2cbf");
  
  // Populate HTML Y-axis labels (skip top label to avoid conflict with legend)
  const labelsContainer = canvas.parentElement?.querySelector(".y-axis-labels");
  if (labelsContainer) {
    labelsContainer.innerHTML = "";
    // Add empty spacer for the top
    const topSpacer = document.createElement("div");
    topSpacer.style.flex = "1";
    labelsContainer.appendChild(topSpacer);
    // Add labels for i=1 to i=4
    for (let i = 1; i <= 4; i += 1) {
      const value = yMax - (i * (yMax - yMin)) / 4;
      const label = formatNumber(value);
      const labelEl = document.createElement("div");
      labelEl.textContent = label;
      labelEl.style.flex = "1";
      labelEl.style.display = "flex";
      labelEl.style.alignItems = "center";
      labelEl.style.justifyContent = "flex-end";
      labelEl.style.paddingRight = "8px";
      labelEl.style.fontSize = "12px";
      labelEl.style.fontWeight = "bold";
      labelEl.style.color = "#1e1f2e";
      labelEl.style.backgroundColor = "rgba(255, 255, 255, 0.9)";
      labelEl.style.borderRadius = "2px";
      labelEl.style.marginRight = "-4px";
      labelsContainer.appendChild(labelEl);
    }
  }
  drawLegend(ctx);

  ctx.fillStyle = "#6b7280";
  ctx.font = "11px IBM Plex Mono";
  const firstLabel = labels[0] ?? "";
  const lastLabel = labels[labels.length - 1] ?? "";
  ctx.fillText(firstLabel, margin.left, height - 8);
  const lastWidth = ctx.measureText(lastLabel).width;
  ctx.fillText(lastLabel, Math.max(margin.left, width - margin.right - lastWidth), height - 8);
}

function renderCard(result) {
  const node = cardTemplate.content.cloneNode(true);
  node.querySelector(".variant").textContent = result.variant.replaceAll("_", " ");
  node.querySelector(".demand-type").textContent = result.demand_type;
  node.querySelector(".assigned-model").textContent = result.selected_model ?? "No model assigned";
  node.querySelector(".wmape").textContent = result.wmape_percent == null ? "n/a" : `${formatNumber(result.wmape_percent)}%`;
  node.querySelector(".wmape-rolling").textContent = result.wmape_3month_rolling == null ? "n/a" : `${formatNumber(result.wmape_3month_rolling)}%`;

  const prescribedEl = node.querySelector(".prescribed");
  if (result.prescribed_forecast) {
    prescribedEl.textContent = `${result.prescribed_forecast.month}  actual=${formatNumber(result.prescribed_forecast.actual_demand)}  forecast=${formatNumber(result.prescribed_forecast.forecast)}`;
  } else {
    prescribedEl.textContent = "No forecast row available for the assigned model and SKU.";
  }

  const historyBody = node.querySelector(".history");
  const futureBody = node.querySelector(".future-history");
  const futureSummary = node.querySelector(".future-summary");
  const futureNote = node.querySelector(".future-note");
  if (!result.forecast_history.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="3">No forecast history found.</td>';
    historyBody.appendChild(row);
  } else {
    result.forecast_history.forEach((point) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${point.month}</td>
        <td>${formatNumber(point.actual_demand)}</td>
        <td>${formatNumber(point.forecast)}</td>
      `;
      historyBody.appendChild(row);
    });
  }

  if (!result.future_predictions.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="2">No future predictions generated.</td>';
    futureBody.appendChild(row);
    futureSummary.textContent = "Future horizon returned: 0 months.";
    futureNote.textContent = "";
  } else {
    result.future_predictions.forEach((point) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${point.month}</td>
        <td>${formatNumber(point.forecast)}</td>
      `;
      futureBody.appendChild(row);
    });
    futureSummary.textContent = `Future horizon returned: ${result.future_predictions.length} months (${result.future_predictions[0].month} to ${result.future_predictions[result.future_predictions.length - 1].month}).`;
    futureNote.textContent = result.future_is_flat
      ? "Model output is flat across horizon (steady-state forecast)."
      : "Model output varies month by month across the 2-month horizon.";
  }

  resultsEl.appendChild(node);
  const canvas = resultsEl.lastElementChild?.querySelector(".line-chart");
  if (canvas) {
    drawChart(canvas, result);
  }
}

async function runSearch() {
  const sku = skuInput.value.trim();
  clearResults();

  if (!sku) {
    statusEl.textContent = "Enter a SKU ID to search.";
    return;
  }

  statusEl.textContent = `Searching ${sku}...`;
  searchBtn.disabled = true;

  try {
    const response = await fetch(`/api/sku/${encodeURIComponent(sku)}`);
    const payload = await response.json();

    if (!response.ok) {
      statusEl.textContent = payload.detail ?? "Search failed.";
      return;
    }

    statusEl.textContent = `Found ${payload.results.length} dataset result(s) for ${payload.sku_id}.`;
    payload.results.forEach(renderCard);
  } catch {
    statusEl.textContent = "Unable to reach backend service.";
  } finally {
    searchBtn.disabled = false;
  }
}

searchBtn.addEventListener("click", runSearch);
skuInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runSearch();
  }
});
