// Exports a portable churn model + precomputed analytics for the Next.js app.
// Mirrors src/preprocess.py (StandardScaler + OneHotEncoder) and trains a
// logistic regression so scoring can run anywhere without Python.
// Run: node scripts/build-model.mjs
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";

const ROOT = new URL("..", import.meta.url);
const NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"];
const CATEGORICAL = [
  "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
  "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
  "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
  "PaperlessBilling", "PaymentMethod",
];

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(header.map((h, i) => [h, (cells[i] ?? "").trim()]));
  });
}

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(42);
const shuffle = (arr) => {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
};

const raw = parseCsv(readFileSync(new URL("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", ROOT), "utf8"));
const rows = raw.map((r) => {
  const tenure = Number(r.tenure);
  const monthly = Number(r.MonthlyCharges);
  const total = r.TotalCharges === "" || Number.isNaN(Number(r.TotalCharges)) ? tenure * monthly : Number(r.TotalCharges);
  return { ...r, SeniorCitizen: Number(r.SeniorCitizen), tenure, MonthlyCharges: monthly, TotalCharges: total, Churn: r.Churn === "Yes" ? 1 : 0 };
});

// Stratified 80/20 split
const pos = shuffle(rows.filter((r) => r.Churn === 1));
const neg = shuffle(rows.filter((r) => r.Churn === 0));
const nPosTest = Math.round(pos.length * 0.2);
const nNegTest = Math.round(neg.length * 0.2);
const test = [...pos.slice(0, nPosTest), ...neg.slice(0, nNegTest)];
const train = shuffle([...pos.slice(nPosTest), ...neg.slice(nNegTest)]);

function fitPreprocessor(data) {
  const numeric = NUMERIC.map((name) => {
    const vals = data.map((r) => r[name]);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length) || 1;
    return { name, mean, std };
  });
  const categorical = CATEGORICAL.map((name) => ({ name, categories: [...new Set(data.map((r) => r[name]))].sort() }));
  return { numeric, categorical };
}

function transform(pre, r) {
  const x = pre.numeric.map((n) => (r[n.name] - n.mean) / n.std);
  for (const c of pre.categorical) for (const cat of c.categories) x.push(r[c.name] === cat ? 1 : 0);
  return x;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

function fitLogistic(X, y, { C = 1, epochs = 3000, lr = 0.1 } = {}) {
  const n = X.length, d = X[0].length;
  const w = new Array(d).fill(0); let b = 0;
  const m = new Array(d + 1).fill(0), v = new Array(d + 1).fill(0);
  for (let t = 1; t <= epochs; t++) {
    const g = new Array(d).fill(0); let gb = 0;
    for (let i = 0; i < n; i++) {
      let z = b; const xi = X[i];
      for (let j = 0; j < d; j++) z += w[j] * xi[j];
      const err = sigmoid(z) - y[i];
      gb += err;
      for (let j = 0; j < d; j++) g[j] += err * xi[j];
    }
    for (let j = 0; j <= d; j++) {
      const grad = j < d ? g[j] / n + w[j] / (C * n) : gb / n;
      m[j] = 0.9 * m[j] + 0.1 * grad;
      v[j] = 0.999 * v[j] + 0.001 * grad * grad;
      const step = (lr * (m[j] / (1 - 0.9 ** t))) / (Math.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8);
      if (j < d) w[j] -= step; else b -= step;
    }
  }
  return { w, b };
}

const predictProba = (model, X) => X.map((x) => sigmoid(x.reduce((s, xj, j) => s + xj * model.w[j], model.b)));

function rocAuc(y, p) {
  const idx = p.map((v, i) => [v, y[i]]).sort((a, b) => a[0] - b[0]);
  let rankSum = 0, nPos = 0;
  for (let i = 0; i < idx.length;) {
    let j = i; while (j < idx.length && idx[j][0] === idx[i][0]) j++;
    const avgRank = (i + j + 1) / 2;
    for (let k = i; k < j; k++) if (idx[k][1] === 1) { rankSum += avgRank; nPos++; }
    i = j;
  }
  const nNeg = y.length - nPos;
  return (rankSum - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

function metricsAt(y, p, threshold) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  p.forEach((v, i) => { const pred = v >= threshold ? 1 : 0; if (pred && y[i]) tp++; else if (pred) fp++; else if (y[i]) fn++; else tn++; });
  const precision = tp / (tp + fp || 1), recall = tp / (tp + fn || 1);
  return {
    threshold, accuracy: (tp + tn) / y.length, precision, recall,
    f1: (2 * precision * recall) / (precision + recall || 1), roc_auc: rocAuc(y, p),
    confusion: { tp, fp, tn, fn },
  };
}

// 5-fold out-of-fold predictions on training data to choose the threshold
const K = 5; const oof = new Array(train.length);
for (let k = 0; k < K; k++) {
  const fitIdx = train.map((_, i) => i).filter((i) => i % K !== k);
  const valIdx = train.map((_, i) => i).filter((i) => i % K === k);
  const pre = fitPreprocessor(fitIdx.map((i) => train[i]));
  const model = fitLogistic(fitIdx.map((i) => transform(pre, train[i])), fitIdx.map((i) => train[i].Churn), { epochs: 800 });
  predictProba(model, valIdx.map((i) => transform(pre, train[i]))).forEach((p, j) => { oof[valIdx[j]] = p; });
}
const yTrain = train.map((r) => r.Churn);
let threshold = 0.5, bestF1 = 0;
for (let t = 0.1; t <= 0.9; t += 0.01) {
  const { f1 } = metricsAt(yTrain, oof, t);
  if (f1 > bestF1) { bestF1 = f1; threshold = Math.round(t * 100) / 100; }
}

const pre = fitPreprocessor(train);
const XTrain = train.map((r) => transform(pre, r));
const model = fitLogistic(XTrain, yTrain);
const yTest = test.map((r) => r.Churn);
const pTest = predictProba(model, test.map((r) => transform(pre, r)));
const round = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, typeof v === "number" ? Math.round(v * 10000) / 10000 : v]));

// Feature means in transformed space -> base value for linear (exact) SHAP
const d = XTrain[0].length;
const featureMeans = Array.from({ length: d }, (_, j) => XTrain.reduce((s, x) => s + x[j], 0) / XTrain.length);

const columnOptions = Object.fromEntries(pre.categorical.map((c) => [c.name, c.categories]));
const exported = {
  model: "LogisticRegression",
  threshold,
  intercept: model.b,
  weights: model.w,
  featureMeans,
  numeric: pre.numeric,
  categorical: pre.categorical,
  columnOptions,
  test_metrics: round(metricsAt(yTest, pTest, threshold)),
  test_metrics_at_0_5: round(metricsAt(yTest, pTest, 0.5)),
  n_train: train.length,
  n_test: test.length,
};

// Global importance: mean |contribution| per original feature on the test set
const importance = {};
for (const r of test) {
  const x = transform(pre, r); let j = 0;
  for (const n of pre.numeric) { importance[n.name] = (importance[n.name] ?? 0) + Math.abs(model.w[j] * (x[j] - featureMeans[j])); j++; }
  for (const c of pre.categorical) {
    let s = 0; for (let k = 0; k < c.categories.length; k++, j++) s += model.w[j] * (x[j] - featureMeans[j]);
    importance[c.name] = (importance[c.name] ?? 0) + Math.abs(s);
  }
}
exported.global_importance = Object.entries(importance)
  .map(([feature, v]) => ({ feature, importance: Math.round((v / test.length) * 10000) / 10000 }))
  .sort((a, b) => b.importance - a.importance);

// ---------------------------------------------------------------- analytics
const rate = (list) => list.reduce((s, r) => s + r.Churn, 0) / list.length;
const by = (key) => {
  const groups = {};
  for (const r of rows) (groups[key(r)] ??= []).push(r);
  return Object.entries(groups).map(([name, list]) => ({ name, churn_rate: rate(list), customers: list.length }))
    .sort((a, b) => b.churn_rate - a.churn_rate);
};
const tenureGroup = (r) => (r.tenure <= 12 ? "0-12" : r.tenure <= 24 ? "13-24" : r.tenure <= 48 ? "25-48" : "49+");
const analytics = {
  kpis: {
    customers: rows.length,
    churn_rate: rate(rows),
    avg_monthly: rows.reduce((s, r) => s + r.MonthlyCharges, 0) / rows.length,
    revenue_at_risk: rows.filter((r) => r.Churn).reduce((s, r) => s + r.MonthlyCharges, 0),
  },
  by_contract: by((r) => r.Contract),
  by_tenure: by(tenureGroup).sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })),
  by_payment: by((r) => r.PaymentMethod),
  by_internet: by((r) => r.InternetService),
  by_tech_support: by((r) => r.TechSupport),
};

// ---------------------------------------------------------------- SQL explorer
const db = new DatabaseSync(":memory:");
const cols = Object.keys(rows[0]);
db.exec(`CREATE TABLE customers (${cols.map((c) => `"${c}"`).join(", ")})`);
const insert = db.prepare(`INSERT INTO customers VALUES (${cols.map(() => "?").join(", ")})`);
db.exec("BEGIN"); for (const r of rows) insert.run(...cols.map((c) => r[c])); db.exec("COMMIT");
const sqlText = readFileSync(new URL("sql/queries.sql", ROOT), "utf8");
const queries = sqlText.split(/^-- name:/m).slice(1).map((block) => {
  const [title, ...body] = block.split("\n");
  const sql = body.join("\n").trim().replace(/;\s*$/, "");
  const result = db.prepare(sql).all();
  return { title: title.trim(), sql, columns: result.length ? Object.keys(result[0]) : [], rows: result.map((r) => Object.values(r)) };
});

const original = JSON.parse(readFileSync(new URL("models/metrics.json", ROOT), "utf8"));

mkdirSync(new URL("lib/generated/", ROOT), { recursive: true });
writeFileSync(new URL("lib/generated/model.json", ROOT), JSON.stringify(exported));
writeFileSync(new URL("lib/generated/analytics.json", ROOT), JSON.stringify(analytics));
writeFileSync(new URL("lib/generated/queries.json", ROOT), JSON.stringify(queries));
writeFileSync(new URL("lib/generated/python-metrics.json", ROOT), JSON.stringify(original, null, 2));
console.log("threshold", threshold, "test", exported.test_metrics);
