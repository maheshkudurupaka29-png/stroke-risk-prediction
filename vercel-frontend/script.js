
// Replace this with your actual Render backend URL
const API_URL = "https://stroke-risk-prediction-api-i77j.onrender.com";

// Get HTML elements
const form = document.getElementById("predictionForm");
const fieldsContainer = document.getElementById("patientFields");
const result = document.getElementById("result");

// Load input fields from FastAPI
async function loadFields() {
    try {
        const response = await fetch(`${API_URL}/schema`);

        if (!response.ok) {
            throw new Error("Unable to load fields");
        }

        const schema = await response.json();
        fieldsContainer.innerHTML = "";

        schema.numeric_features.forEach(function(feature) {
            const label = document.createElement("label");
            label.textContent = feature;

            const input = document.createElement("input");
            input.type = "number";
            input.step = "any";
            input.name = feature;
            input.required = true;

            fieldsContainer.appendChild(label);
            fieldsContainer.appendChild(input);
            fieldsContainer.appendChild(document.createElement("br"));
        });

        schema.categorical_features.forEach(function(feature) {
            const label = document.createElement("label");
            label.textContent = feature;

            const input = document.createElement("input");
            input.type = "text";
            input.name = feature;
            input.required = true;

            fieldsContainer.appendChild(label);
            fieldsContainer.appendChild(input);
            fieldsContainer.appendChild(document.createElement("br"));
        });

    } catch (error) {
        fieldsContainer.textContent =
            "Error loading fields: " + error.message;
    }
}

// Send entered values to FastAPI
form.addEventListener("submit", async function(event) {
    event.preventDefault();

    const inputData = {};

    const inputs = fieldsContainer.querySelectorAll("input");

    inputs.forEach(function(input) {
        inputData[input.name] =
            input.type === "number"
                ? Number(input.value)
                : input.value.trim();
    });

    try {
        result.textContent = "Predicting...";

        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ data: inputData })
        });

        const prediction = await response.json();

        if (!response.ok) {
            throw new Error(JSON.stringify(prediction));
        }

        result.textContent =
            "Prediction Result: " + JSON.stringify(prediction);

    } catch (error) {
        result.textContent = "Prediction failed: " + error.message;
    }
});

// Run when the website opens
loadFields();
