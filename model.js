
class LinearRegression {
    constructor() {
        this.weights = [0, 0, 0];
        this.bias = 0;
        this.means = [];
        this.stds = [];
        this.trained = false;
    }

    // Calculate the average of an array
    mean(values) {
        return values.reduce((sum, value) => sum + value, 0)
            / values.length;
    }

    // Calculate standard deviation
    standardDeviation(values, mean) {
        const variance = values.reduce(
            (sum, value) => sum + (value - mean) ** 2,
            0
        ) / values.length;

        return Math.sqrt(variance) || 1;
    }

    // Standardize features so they are on comparable scales
    standardize(features) {
        return features.map((row) =>
            row.map((value, j) =>
                (value - this.means[j]) / this.stds[j]
            )
        );
    }

    // Train the model using gradient descent
    fit(X, y, learningRate = 0.03, iterations = 5000) {
        if (!X.length || X.length !== y.length) {
            throw new Error("Invalid training data.");
        }

        if (X.some(row =>
            row.length !== 3 ||
            row.some(value => !Number.isFinite(value))
        ) || y.some(value => !Number.isFinite(value))) {
            throw new Error("Training data contains invalid values.");
        }

        const featureCount = X[0].length;

        // Calculate feature means and standard deviations
        this.means = [];
        this.stds = [];

        for (let j = 0; j < featureCount; j++) {
            const column = X.map(row => row[j]);
            const columnMean = this.mean(column);

            this.means.push(columnMean);
            this.stds.push(
                this.standardDeviation(column, columnMean)
            );
        }

        const normalizedX = this.standardize(X);

        // Initialize parameters
        this.weights = new Array(featureCount).fill(0);
        this.bias = this.mean(y);

        const sampleCount = X.length;

        // Gradient descent optimization
        for (let iteration = 0; iteration < iterations; iteration++) {
            const weightGradients =
                new Array(featureCount).fill(0);

            let biasGradient = 0;

            for (let i = 0; i < sampleCount; i++) {
                let prediction = this.bias;

                for (let j = 0; j < featureCount; j++) {
                    prediction +=
                        this.weights[j] * normalizedX[i][j];
                }

                const error = prediction - y[i];

                biasGradient += error;

                for (let j = 0; j < featureCount; j++) {
                    weightGradients[j] +=
                        error * normalizedX[i][j];
                }
            }

            // Update parameters
            this.bias -= learningRate *
                (biasGradient / sampleCount);

            for (let j = 0; j < featureCount; j++) {
                this.weights[j] -= learningRate *
                    (weightGradients[j] / sampleCount);
            }
        }

        this.trained = true;
    }

    // Predict the price of one house
    predict(features) {
        if (!this.trained) {
            throw new Error("Train the model before predicting.");
        }

        if (
            !Array.isArray(features) ||
            features.length !== 3 ||
            features.some(value => !Number.isFinite(value))
        ) {
            throw new Error("Enter valid values for all three features.");
        }

        const normalized = features.map(
            (value, j) => (value - this.means[j]) / this.stds[j]
        );

        let prediction = this.bias;

        for (let j = 0; j < this.weights.length; j++) {
            prediction += this.weights[j] * normalized[j];
        }

        return Math.max(0, prediction);
    }

    // Calculate R-squared on supplied data
    score(X, y) {
        const actualMean = this.mean(y);

        const totalVariation = y.reduce(
            (sum, actual) => sum + (actual - actualMean) ** 2,
            0
        );

        if (totalVariation === 0) {
            return null;
        }

        const residualVariation = X.reduce(
            (sum, features, i) =>
                sum + (y[i] - this.predict(features)) ** 2,
            0
        );

        return 1 - residualVariation / totalVariation;
    }
}
