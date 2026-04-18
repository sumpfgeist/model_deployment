let bundle = null;

function $(id) {
  return document.getElementById(id);
}

function formatMoney(value) {
  if (!Number.isFinite(value)) return "$—";
  return "$" + value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function updateSliderLabels() {
  $("age_val").textContent = $("age").value;
  $("bmi_val").textContent = $("bmi").value;
  $("children_val").textContent = $("children").value;
}

function standardize(value, mean, scale) {
  return (value - mean) / scale;
}

function readInputs() {
  return {
    age: Number($("age").value),
    bmi: Number($("bmi").value),
    children: Number($("children").value),
    sex: $("sex").value,
    smoker: $("smoker").value,
    region: $("region").value,
  };
}

function buildFeatureDict(input, preprocess) {
  const featureDict = {};

  // Numeric features
  preprocess.numeric_features.forEach((feature, i) => {
    featureDict[feature] = standardize(
      Number(input[feature]),
      preprocess.numeric_mean[i],
      preprocess.numeric_scale[i]
    );
  });

  // One-hot encoded categorical features
  preprocess.categorical_features.forEach((feature) => {
    const categories = preprocess.categorical_categories[feature];
    categories.forEach((cat) => {
      const key = `${feature}_${cat}`;
      featureDict[key] = input[feature] === cat ? 1 : 0;
    });
  });

  return featureDict;
}

function dictToVector(featureNamesOut, featureDict) {
  return featureNamesOut.map((name) => (name in featureDict ? featureDict[name] : 0));
}

function predict(input) {
  const preprocess = bundle.preprocess;
  const featureDict = buildFeatureDict(input, preprocess);
  const x = dictToVector(preprocess.feature_names_out, featureDict);

  // Use ml.js Matrix for the linear algebra
  const xMat = new ML.Matrix([x]); // 1 x n
  const coefMat = ML.Matrix.columnVector(bundle.model.coef); // n x 1
  const yMat = xMat.mmul(coefMat); // 1 x 1

  return yMat.get(0, 0) + bundle.model.intercept;
}

function setStatus(message, type = "info") {
  const box = $("statusBox");
  box.className = `alert alert-${type}`;
  box.textContent = message;
}

function renderMetrics() {
  const m = bundle.meta.metrics_test;
  $("metricsBox").textContent =
    `MAE: ${m.MAE.toFixed(2)} | RMSE: ${m.RMSE.toFixed(2)} | R²: ${m.R2.toFixed(4)}`;

  $("residualImg").src = "./residuals.png";
  $("pvaImg").src = "./pred_vs_actual.png";
}

async function loadModel() {
  const response = await fetch("./model.json");
  if (!response.ok) {
    throw new Error(`Could not load model.json (${response.status})`);
  }
  return await response.json();
}

async function init() {
  updateSliderLabels();

  ["age", "bmi", "children"].forEach((id) => {
    $(id).addEventListener("input", updateSliderLabels);
  });

  try {
    bundle = await loadModel();
    setStatus("Model loaded successfully.", "success");
    renderMetrics();
  } catch (error) {
    console.error(error);
    setStatus("Failed to load model.", "danger");
    return;
  }

  $("predictBtn").addEventListener("click", () => {
    try {
      const input = readInputs();
      const prediction = predict(input);
      $("predValue").textContent = formatMoney(prediction);
    } catch (error) {
      console.error(error);
      setStatus("Prediction failed.", "danger");
    }
  });
}

init();