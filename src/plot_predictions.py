import json
import os

import pandas as pd
from xgboost import XGBClassifier


# =========================================================
# Load data
# =========================================================

df = pd.read_csv(
    "data/aapl.csv",
    index_col="Date",
    parse_dates=True,
)


# =========================================================
# Feature engineering
# =========================================================

df["return_1d"] = df["Adj Close"].pct_change()
df["return_5d"] = df["Adj Close"].pct_change(5)
df["return_10d"] = df["Adj Close"].pct_change(10)
df["return_20d"] = df["Adj Close"].pct_change(20)

df["volatility_20d"] = df["return_1d"].rolling(20).std()

df["volume_ratio_20d"] = (
    df["Volume"] / df["Volume"].rolling(20).mean()
)

df["ma_10"] = df["Adj Close"].rolling(10).mean()
df["ma_20"] = df["Adj Close"].rolling(20).mean()

df["distance_ma_10"] = (
    df["Adj Close"] / df["ma_10"] - 1
)

df["distance_ma_20"] = (
    df["Adj Close"] / df["ma_20"] - 1
)

df["high_low_range"] = (
    (df["High"] - df["Low"]) / df["Adj Close"]
)


# =========================================================
# Target
# =========================================================

df["future_return_1d"] = (
    df["Adj Close"]
    .pct_change()
    .shift(-1)
)

df["target"] = (
    df["future_return_1d"] > 0
).astype(int)


# =========================================================
# Features
# =========================================================

features = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_20d",
    "volume_ratio_20d",
    "distance_ma_10",
    "distance_ma_20",
    "high_low_range",
]

df = df.dropna(
    subset=features + ["future_return_1d"]
).copy()


# =========================================================
# Model configuration
# =========================================================

params = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "random_state": 42,
    "eval_metric": "logloss",
}

validation_years = [
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
]


# =========================================================
# Walk-forward predictions
# =========================================================

all_predictions = []

for validation_year in validation_years:

    validation_df = df.loc[
        df.index.year == validation_year
    ].copy()

    if validation_df.empty:
        continue

    validation_start = validation_df.index.min()

    train_df = df.loc[
        df.index < validation_start
    ].copy()

    # The final training row has a next-day target that
    # touches the first validation day, so remove it.
    if not train_df.empty:
        train_df = train_df.iloc[:-1]

    X_train = train_df[features]
    y_train = train_df["target"]
    X_val = validation_df[features]

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    probability_up = model.predict_proba(X_val)[:, 1]

    prediction = (
        probability_up >= 0.5
    ).astype(int)

    result = validation_df[
        [
            "Adj Close",
            "future_return_1d",
            "target",
        ]
    ].copy()

    result["probability_up"] = probability_up
    result["prediction"] = prediction

    result["correct"] = (
        result["prediction"]
        == result["target"]
    )

    all_predictions.append(result)


if not all_predictions:
    raise RuntimeError(
        "No walk-forward predictions were generated."
    )

predictions_df = pd.concat(all_predictions)


# =========================================================
# Prepare data for browser
# =========================================================

records = []
running_correct = 0

for i, (date, row) in enumerate(
    predictions_df.iterrows(),
    start=1,
):

    is_correct = bool(row["correct"])

    if is_correct:
        running_correct += 1

    running_accuracy = (
        running_correct / i
    )

    records.append(
        {
            "date": date.strftime("%Y-%m-%d"),
            "price": round(
                float(row["Adj Close"]),
                2,
            ),
            "probability_up": round(
                float(row["probability_up"]) * 100,
                2,
            ),
            "prediction": (
                "UP"
                if int(row["prediction"]) == 1
                else "DOWN"
            ),
            "actual": (
                "UP"
                if int(row["target"]) == 1
                else "DOWN"
            ),
            "future_return": round(
                float(row["future_return_1d"]) * 100,
                3,
            ),
            "correct": is_correct,
            "running_accuracy": round(
                running_accuracy * 100,
                2,
            ),
        }
    )


overall_accuracy = float(
    predictions_df["correct"].mean()
)

always_up_baseline = float(
    predictions_df["target"].mean()
)

print(
    f"Walk-forward accuracy: "
    f"{overall_accuracy:.2%}"
)

print(
    f"Always-up baseline: "
    f"{always_up_baseline:.2%}"
)


# =========================================================
# Generate animated HTML
# =========================================================

data_json = json.dumps(records)

html_template = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>
        AAPL Walk-Forward Prediction Demo
    </title>

    <script
        src="https://cdn.plot.ly/plotly-2.35.2.min.js">
    </script>

    <style>
        body {
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;

            margin: 32px;
            background: #f7f7f7;
        }

        h1 {
            margin-bottom: 6px;
        }

        .subtitle {
            color: #666;
            margin-bottom: 20px;
        }

        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }

        button,
        select {
            min-height: 42px;
            padding: 8px 16px;
            font-size: 14px;
        }

        button {
            cursor: pointer;
        }

        .status {
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }

        .card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow:
                0 1px 4px
                rgba(0, 0, 0, 0.12);
        }

        .label {
            font-size: 12px;
            color: #777;
            margin-bottom: 4px;
        }

        .value {
            font-size: 21px;
            font-weight: 600;
        }

        #chart {
            background: white;
            border-radius: 8px;
        }

        .correct {
            color: #188038;
        }

        .wrong {
            color: #d93025;
        }
    </style>
</head>

<body>

    <h1>
        AAPL XGBoost Walk-Forward Prediction
    </h1>

    <div class="subtitle">
        The model predicts whether AAPL will rise on the
        next trading day. Every displayed prediction is
        out-of-sample.
    </div>

    <div class="controls">

        <button id="playBtn" type="button">
            ▶ Play
        </button>

        <button id="pauseBtn" type="button">
            ⏸ Pause
        </button>

        <button id="resetBtn" type="button">
            ↺ Reset
        </button>

        <label>
            Speed:

            <select id="speedSelect">
                <option value="500">
                    1×
                </option>

                <option value="100">
                    5×
                </option>

                <option value="25">
                    20×
                </option>
            </select>
        </label>

    </div>

    <div class="status">

        <div class="card">
            <div class="label">
                Date
            </div>
            <div
                id="dateValue"
                class="value">
                -
            </div>
        </div>

        <div class="card">
            <div class="label">
                AAPL Price
            </div>
            <div
                id="priceValue"
                class="value">
                -
            </div>
        </div>

        <div class="card">
            <div class="label">
                Probability Up
            </div>
            <div
                id="probabilityValue"
                class="value">
                -
            </div>
        </div>

        <div class="card">
            <div class="label">
                Prediction
            </div>
            <div
                id="predictionValue"
                class="value">
                -
            </div>
        </div>

        <div class="card">
            <div class="label">
                Actual Next Day
            </div>
            <div
                id="actualValue"
                class="value">
                -
            </div>
        </div>

        <div class="card">
            <div class="label">
                Result
            </div>
            <div
                id="resultValue"
                class="value">
                -
            </div>
        </div>

        <div class="card">
            <div class="label">
                Running Accuracy
            </div>
            <div
                id="accuracyValue"
                class="value">
                -
            </div>
        </div>

    </div>

    <div id="chart"></div>


    <script>
        const data = __DATA_JSON__;

        let currentIndex = 0;
        let timer = null;

        const priceTrace = {
            x: [],
            y: [],
            mode: "lines",
            name: "AAPL Price",
            xaxis: "x",
            yaxis: "y",
            hovertemplate:
                "<b>%{x}</b>" +
                "<br>AAPL: $%{y:.2f}" +
                "<extra></extra>"
        };

        const probabilityTrace = {
            x: [],
            y: [],
            mode: "lines+markers",
            name: "Probability Up",
            xaxis: "x2",
            yaxis: "y2",
            hovertemplate:
                "<b>%{x}</b>" +
                "<br>Probability up: %{y:.1f}%" +
                "<extra></extra>"
        };

        const thresholdTrace = {
            x: [],
            y: [],
            mode: "lines",
            name: "50% Threshold",
            line: {
                dash: "dash"
            },
            xaxis: "x2",
            yaxis: "y2",
            hoverinfo: "skip"
        };

        const layout = {
            height: 800,

            title: {
                text:
                    "Walk-Forward Prediction Playback" +
                    "<br><sup>" +
                    "Overall accuracy: " +
                    "__OVERALL_ACCURACY__" +
                    " | Always-up baseline: " +
                    "__BASELINE__" +
                    "</sup>"
            },

            grid: {
                rows: 2,
                columns: 1,
                pattern: "independent"
            },

            xaxis: {
                title: "Date"
            },

            yaxis: {
                title: "AAPL Price ($)"
            },

            xaxis2: {
                title: "Date"
            },

            yaxis2: {
                title: "Probability Up (%)",
                range: [0, 100]
            },

            hovermode: "x unified",

            margin: {
                l: 70,
                r: 30,
                t: 90,
                b: 60
            }
        };

        Plotly.newPlot(
            "chart",
            [
                priceTrace,
                probabilityTrace,
                thresholdTrace
            ],
            layout,
            {
                responsive: true,
                displaylogo: false
            }
        );

        function updateStatus(point) {

            document.getElementById(
                "dateValue"
            ).innerText = point.date;

            document.getElementById(
                "priceValue"
            ).innerText =
                "$" + point.price.toFixed(2);

            document.getElementById(
                "probabilityValue"
            ).innerText =
                point.probability_up.toFixed(1) + "%";

            document.getElementById(
                "predictionValue"
            ).innerText =
                point.prediction;

            document.getElementById(
                "actualValue"
            ).innerText =
                point.actual;

            const resultElement =
                document.getElementById(
                    "resultValue"
                );

            if (point.correct) {
                resultElement.innerText =
                    "Correct ✓";

                resultElement.className =
                    "value correct";
            } else {
                resultElement.innerText =
                    "Wrong ✕";

                resultElement.className =
                    "value wrong";
            }

            document.getElementById(
                "accuracyValue"
            ).innerText =
                point.running_accuracy.toFixed(1)
                + "%";
        }


        function renderPoint(index) {

            if (index >= data.length) {
                pause();
                return;
            }

            const point = data[index];

            Plotly.extendTraces(
                "chart",
                {
                    x: [
                        [point.date],
                        [point.date],
                        [point.date]
                    ],
                    y: [
                        [point.price],
                        [point.probability_up],
                        [50]
                    ]
                },
                [0, 1, 2]
            );

            updateStatus(point);

            currentIndex += 1;
        }


        function scheduleNextPoint() {

            if (currentIndex >= data.length) {
                pause();
                return;
            }

            const speed = Number(
                document.getElementById(
                    "speedSelect"
                ).value
            );

            timer = setTimeout(
                function () {
                    timer = null;
                    renderPoint(currentIndex);
                    scheduleNextPoint();
                },
                speed
            );
        }


        function play() {

            if (timer !== null) {
                return;
            }

            if (currentIndex >= data.length) {
                resetPlot();
            }

            if (currentIndex === 0) {
                renderPoint(currentIndex);
            }

            scheduleNextPoint();
        }


        function pause() {

            if (timer !== null) {
                clearTimeout(timer);
                timer = null;
            }
        }


        function resetPlot() {

            pause();

            currentIndex = 0;

            Plotly.react(
                "chart",
                [
                    {
                        ...priceTrace,
                        x: [],
                        y: []
                    },
                    {
                        ...probabilityTrace,
                        x: [],
                        y: []
                    },
                    {
                        ...thresholdTrace,
                        x: [],
                        y: []
                    }
                ],
                layout,
                {
                    responsive: true,
                    displaylogo: false
                }
            );

            document.getElementById(
                "dateValue"
            ).innerText = "-";

            document.getElementById(
                "priceValue"
            ).innerText = "-";

            document.getElementById(
                "probabilityValue"
            ).innerText = "-";

            document.getElementById(
                "predictionValue"
            ).innerText = "-";

            document.getElementById(
                "actualValue"
            ).innerText = "-";

            const resultElement =
                document.getElementById(
                    "resultValue"
                );

            resultElement.innerText = "-";
            resultElement.className = "value";

            document.getElementById(
                "accuracyValue"
            ).innerText = "-";
        }


        document.getElementById(
            "playBtn"
        ).addEventListener(
            "click",
            play
        );

        document.getElementById(
            "pauseBtn"
        ).addEventListener(
            "click",
            pause
        );

        document.getElementById(
            "resetBtn"
        ).addEventListener(
            "click",
            resetPlot
        );
    </script>

</body>
</html>
"""

html = (
    html_template
    .replace(
        "__DATA_JSON__",
        data_json,
    )
    .replace(
        "__OVERALL_ACCURACY__",
        f"{overall_accuracy:.1%}",
    )
    .replace(
        "__BASELINE__",
        f"{always_up_baseline:.1%}",
    )
)


# =========================================================
# Save output
# =========================================================

os.makedirs(
    "outputs",
    exist_ok=True,
)

output_file = (
    "outputs/"
    "aapl_walkforward_predictions.html"
)

with open(
    output_file,
    "w",
    encoding="utf-8",
) as file:
    file.write(html)

print(
    "\nSaved animated prediction demo to:"
)

print(output_file)
