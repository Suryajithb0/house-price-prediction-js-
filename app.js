
const model = new LinearRegression();

let trainingData = [];
let priceChart = null;

// Get HTML elements
const form = document.getElementById("prediction-form");

const sizeInput = document.getElementById("size");
const bedroomsInput = document.getElementById("bedrooms");
const ageInput = document.getElementById("age");

const sizeValue = document.getElementById("size-value");
const bedroomsValue = document.getElementById("bedrooms-value");
const ageValue = document.getElementById("age-value");

const predictButton = document.getElementById("predict-button");
const modelStatus = document.getElementById("model-status");

const resultPlaceholder =
    document.getElementById("result-placeholder");

const resultContent =
    document.getElementById("result-content");

const predictedPrice =
    document.getElementById("predicted-price");

const errorMessage =
    document.getElementById("error-message");

const chartNote = document.getElementById("chart-note");

// ==========================================
// 1. UPDATE SLIDER LABELS
// ==========================================

function updateSliderLabels() {
    sizeValue.textContent = `${sizeInput.value} sqft`;

    const bedrooms = Number(bedroomsInput.value);
    bedroomsValue.textContent =
        `${bedrooms} ${bedrooms === 1 ? "bedroom" : "bedrooms"}`;

    const age = Number(ageInput.value);
    ageValue.textContent =
        `${age} ${age === 1 ? "year" : "years"}`;
}

sizeInput.addEventListener("input", updateSliderLabels);
bedroomsInput.addEventListener("input", updateSliderLabels);
ageInput.addEventListener("input", updateSliderLabels);

// ==========================================
// 2. TRAIN-TEST SPLIT
// JavaScript equivalent of sklearn's
// train_test_split(X, y, test_size=0.2)
// ==========================================

function trainTestSplit(X, y, testSize = 0.2) {
    if (X.length !== y.length || X.length < 2) {
        throw new Error("Invalid dataset for train-test split.");
    }

    if (testSize <= 0 || testSize >= 1) {
        throw new Error("testSize must be between 0 and 1.");
    }

    // Keep each input row paired with its actual target
    const indices = X.map((_, i) => i);

    // Shuffle indices using Fisher-Yates
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));

        [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const testCount = Math.max(
        1,
        Math.min(X.length - 1, Math.ceil(X.length * testSize))
    );

    const testIndices = indices.slice(0, testCount);
    const trainIndices = indices.slice(testCount);

    const X_train = trainIndices.map(i => X[i]);
    const X_test = testIndices.map(i => X[i]);

    const y_train = trainIndices.map(i => y[i]);
    const y_test = testIndices.map(i => y[i]);

    return {
        X_train,
        X_test,
        y_train,
        y_test
    };
}

// ==========================================
// 3. EVALUATION METRICS
// ==========================================

function calculateMetrics(y_test, y_pred) {
    const errors = y_test.map(
        (actual, i) => actual - y_pred[i]
    );

    const mae = errors.reduce(
        (sum, error) => sum + Math.abs(error), 0
    ) / errors.length;

    const mse = errors.reduce(
        (sum, error) => sum + error ** 2, 0
    ) / errors.length;

    const rmse = Math.sqrt(mse);

    const meanActual = y_test.reduce(
        (sum, value) => sum + value, 0
    ) / y_test.length;

    const totalVariation = y_test.reduce(
        (sum, value) => sum + (value - meanActual) ** 2, 0
    );

    const residualVariation = errors.reduce(
        (sum, error) => sum + error ** 2, 0
    );

    const r2 = totalVariation === 0
        ? null
        : 1 - residualVariation / totalVariation;

    return { mae, mse, rmse, r2 };
}

// ==========================================
// 4. LOAD DATASET, SPLIT, TRAIN AND TEST
// ==========================================

async function loadDataset() {
    predictButton.disabled = true;

    try {
        modelStatus.textContent = "Loading dataset...";

        const response = await fetch("house_prices.csv");

        if (!response.ok) {
            throw new Error(
                "Could not load house_prices.csv. Check the filename and server."
            );
        }

        const csvText = await response.text();

        const lines = csvText
            .trim()
            .split(/\r?\n/);

        if (lines.length < 6) {
            throw new Error(
                "The dataset needs at least five data records."
            );
        }

        // Read CSV headers
        const headers = lines[0]
            .split(",")
            .map(header => header.trim());

        const requiredHeaders = [
            "size_sqft",
            "bedrooms",
            "age_years",
            "price_lakhs"
        ];

        for (const header of requiredHeaders) {
            if (!headers.includes(header)) {
                throw new Error(`Missing CSV column: ${header}`);
            }
        }

        const columnIndex = Object.fromEntries(
            requiredHeaders.map(header => [
                header,
                headers.indexOf(header)
            ])
        );

        // Parse CSV records
        trainingData = lines
            .slice(1)
            .filter(line => line.trim() !== "")
            .map(line => {
                const values = line.split(",")
                    .map(value => value.trim());

                if (values.length !== headers.length) {
                    throw new Error(`Invalid CSV row: ${line}`);
                }

                const row = requiredHeaders.map(header =>
                    Number(values[columnIndex[header]])
                );

                if (
                    row.some(value => !Number.isFinite(value)) ||
                    row[0] <= 0 ||
                    row[1] <= 0 ||
                    row[2] < 0 ||
                    row[3] <= 0
                ) {
                    throw new Error(`Invalid CSV row: ${line}`);
                }

                return {
                    size: row[0],
                    bedrooms: row[1],
                    age: row[2],
                    price: row[3]
                };
            });

        if (trainingData.length < 5) {
            throw new Error(
                "At least five valid records are required."
            );
        }

        // --------------------------------------
        // STEP A: Separate X (features) and y (target)
        // --------------------------------------

        const X = trainingData.map(house => [
            house.size,
            house.bedrooms,
            house.age
        ]);

        const y = trainingData.map(house => house.price);

        // --------------------------------------
        // STEP B: Split into training and test sets
        // --------------------------------------

        const {
            X_train,
            X_test,
            y_train,
            y_test
        } = trainTestSplit(X, y, 0.2);

        // --------------------------------------
        // STEP C: Train using training data ONLY
        // --------------------------------------

        model.fit(X_train, y_train);

        // --------------------------------------
        // STEP D: Predict using unseen test inputs
        // --------------------------------------

        const y_pred = X_test.map(features =>
            model.predict(features)
        );

        // --------------------------------------
        // STEP E: Evaluate on test data
        // --------------------------------------

        const metrics = calculateMetrics(y_test, y_pred);

        console.log("===== HOUSE PRICE MODEL =====");
        console.log("Total records:", X.length);
        console.log("Training records:", X_train.length);
        console.log("Testing records:", X_test.length);

        console.log("X_train:", X_train);
        console.log("y_train:", y_train);
        console.log("X_test:", X_test);
        console.log("y_test:", y_test);
        console.log("y_pred:", y_pred);

        console.log("MAE:", metrics.mae.toFixed(2), "lakhs");
        console.log("MSE:", metrics.mse.toFixed(2));
        console.log("RMSE:", metrics.rmse.toFixed(2), "lakhs");

        console.log(
            "R²:",
            metrics.r2 === null
                ? "Undefined"
                : metrics.r2.toFixed(3)
        );

        console.table(
            X_test.map((features, i) => ({
                size_sqft: features[0],
                bedrooms: features[1],
                age_years: features[2],
                actual_price: y_test[i],
                predicted_price: Number(y_pred[i].toFixed(2)),
                absolute_error: Number(
                    Math.abs(y_test[i] - y_pred[i]).toFixed(2)
                )
            }))
        );

        // Update status and metrics
        modelStatus.textContent =
            `Trained · ${X_train.length} training records`;

        chartNote.textContent =
            `Test records: ${X_test.length} | ` +
            `MAE: ₹${metrics.mae.toFixed(2)} lakhs | ` +
            `RMSE: ₹${metrics.rmse.toFixed(2)} lakhs | ` +
            `R²: ${metrics.r2 === null
                ? "N/A"
                : metrics.r2.toFixed(3)}.`;

        drawHistoricalChart();

        predictButton.disabled = false;

    } catch (error) {
        console.error("Dataset error:", error);

        modelStatus.textContent = "Model not ready";
        chartNote.textContent = "Unable to load or train the model.";

        showError(error.message);
    }
}

// ==========================================
// 5. HISTORICAL DATA CHART
// ==========================================

function drawHistoricalChart() {
    const canvas = document.getElementById("price-chart");

    if (typeof Chart === "undefined") {
        chartNote.textContent += " Chart.js could not load.";
        return;
    }

    if (priceChart) {
        priceChart.destroy();
    }

    const points = trainingData
        .map(house => ({
            x: house.size,
            y: house.price
        }))
        .sort((a, b) => a.x - b.x);

    priceChart = new Chart(canvas, {
        type: "scatter",

        data: {
            datasets: [{
                label: "House training data",
                data: points,
                backgroundColor: "#a78bfa",
                borderColor: "#c4b5fd",
                pointRadius: 5,
                pointHoverRadius: 8
            }]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    labels: {
                        color: "#cbd5e1"
                    }
                },

                tooltip: {
                    callbacks: {
                        label: context =>
                            `${context.parsed.x} sqft · ` +
                            `₹${context.parsed.y} lakhs`
                    }
                }
            },

            scales: {
                x: {
                    type: "linear",
                    title: {
                        display: true,
                        text: "House size (sqft)",
                        color: "#94a3bd"
                    },
                    ticks: { color: "#94a3bd" },
                    grid: {
                        color: "rgba(148, 163, 189, 0.12)"
                    }
                },

                y: {
                    title: {
                        display: true,
                        text: "Price (₹ lakhs)",
                        color: "#94a3bd"
                    },
                    ticks: { color: "#94a3bd" },
                    grid: {
                        color: "rgba(148, 163, 189, 0.12)"
                    }
                }
            }
        }
    });
}

// ==========================================
// 6. ERROR HANDLING
// ==========================================

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}

function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}

// ==========================================
// 7. PREDICT HOUSE PRICE
// ==========================================

form.addEventListener("submit", function (event) {
    event.preventDefault();

    clearError();

    try {
        if (!model.trained) {
            throw new Error("The model is not ready.");
        }

        const size = Number(sizeInput.value);
        const bedrooms = Number(bedroomsInput.value);
        const age = Number(ageInput.value);

        const prediction = model.predict([
            size,
            bedrooms,
            age
        ]);

        if (!Number.isFinite(prediction)) {
            throw new Error("Invalid prediction returned by the model.");
        }

        predictedPrice.textContent = prediction.toFixed(2);

        document.getElementById("summary-size").textContent =
            `${size} sqft`;

        document.getElementById("summary-bedrooms").textContent =
            `${bedrooms}`;

        document.getElementById("summary-age").textContent =
            `${age} ${age === 1 ? "year" : "years"}`;

        resultPlaceholder.classList.add("hidden");
        resultContent.classList.remove("hidden");

    } catch (error) {
        console.error("Prediction error:", error);
        showError(error.message);
    }
});

// ==========================================
// 8. INITIALIZE APPLICATION
// ==========================================

updateSliderLabels();
loadDataset();
