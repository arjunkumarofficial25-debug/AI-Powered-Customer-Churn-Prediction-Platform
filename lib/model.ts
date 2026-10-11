import modelJson from "@/lib/generated/model.json";

export const NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] as const;
export const CATEGORICAL = [
  "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
  "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
  "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
  "PaperlessBilling", "PaymentMethod",
] as const;
export const FEATURES = [...NUMERIC, ...CATEGORICAL];

export type Customer = Record<string, string | number>;
export type RiskLevel = "High" | "Medium" | "Low";

export const model = modelJson as {
  threshold: number;
  intercept: number;
  weights: number[];
  featureMeans: number[];
  numeric: { name: string; mean: number; std: number }[];
  categorical: { name: string; categories: string[] }[];
  columnOptions: Record<string, string[]>;
  test_metrics: Record<string, number | object>;
  test_metrics_at_0_5: Record<string, number | object>;
  global_importance: { feature: string; importance: number }[];
  n_train: number;
  n_test: number;
};

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

function transform(row: Customer) {
  const x = model.numeric.map((n) => (Number(row[n.name]) - n.mean) / n.std);
  for (const c of model.categorical) for (const cat of c.categories) x.push(String(row[c.name]) === cat ? 1 : 0);
  return x;
}

export function cleanRow(row: Record<string, string>): Customer {
  const out: Customer = {};
  for (const [k, v] of Object.entries(row)) out[k.trim()] = typeof v === "string" ? v.trim() : v;
  for (const n of ["SeniorCitizen", "tenure", "MonthlyCharges"]) out[n] = Number(out[n]);
  const total = Number(out.TotalCharges);
  out.TotalCharges = out.TotalCharges === "" || Number.isNaN(total) ? Number(out.tenure) * Number(out.MonthlyCharges) || 0 : total;
  return out;
}

export function predict(row: Customer) {
  const x = transform(row);
  const z = x.reduce((s, xj, j) => s + xj * model.weights[j], model.intercept);
  return sigmoid(z);
}

/** Exact per-feature contributions (log-odds) for a linear model: w * (x - mean). */
export function explain(row: Customer) {
  const x = transform(row);
  const contributions: { feature: string; value: string | number; contribution: number }[] = [];
  let j = 0;
  for (const n of model.numeric) {
    contributions.push({ feature: n.name, value: row[n.name], contribution: model.weights[j] * (x[j] - model.featureMeans[j]) });
    j++;
  }
  for (const c of model.categorical) {
    let s = 0;
    for (let k = 0; k < c.categories.length; k++, j++) s += model.weights[j] * (x[j] - model.featureMeans[j]);
    contributions.push({ feature: c.name, value: row[c.name], contribution: s });
  }
  return contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}

export function riskLevel(p: number, threshold = model.threshold): RiskLevel {
  if (p >= threshold) return "High";
  if (p >= 0.6 * threshold) return "Medium";
  return "Low";
}

export function recommendations(row: Customer) {
  const tips: string[] = [];
  if (row.Contract === "Month-to-month") tips.push("Offer a discount to move to a 1- or 2-year contract.");
  if (Number(row.tenure) < 12) tips.push("New customer: trigger an onboarding call and early-life engagement offers.");
  if (row.PaymentMethod === "Electronic check") tips.push("Encourage automatic payment (card / bank transfer) with a small credit.");
  if (row.InternetService === "Fiber optic" && row.TechSupport === "No") tips.push("Offer a free tech-support trial for fiber customers.");
  if (row.OnlineSecurity === "No" && row.InternetService !== "No") tips.push("Bundle online security / backup at a promotional price.");
  if (Number(row.MonthlyCharges) > 80) tips.push("High bill: review the plan and propose a cheaper bundle.");
  return tips.length ? tips : ["No specific risk drivers found: keep the customer engaged as usual."];
}

export function validateRows(rows: Record<string, string>[]) {
  if (!rows.length) return ["The file has no data rows."];
  const columns = Object.keys(rows[0]).map((c) => c.trim());
  const missing = FEATURES.filter((f) => !columns.includes(f));
  if (missing.length) return [`Missing required columns: ${missing.join(", ")}`];
  const empty: Record<string, number> = {};
  for (const r of rows) {
    const cleaned = cleanRow(r);
    for (const f of FEATURES) {
      const v = cleaned[f];
      if (v === "" || v === undefined || (typeof v === "number" && Number.isNaN(v))) empty[f] = (empty[f] ?? 0) + 1;
    }
  }
  const entries = Object.entries(empty);
  return entries.length ? [`Empty or invalid values found in: ${entries.map(([k, n]) => `${k} (${n})`).join(", ")}`] : [];
}

export function parseCsv(text: string): Record<string, string>[] {
  const records: string[][] = [];
  let field = "", row: string[] = [], inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') inQuotes = false;
      else field += ch;
    } else if (ch === '"') inQuotes = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((c) => c !== "")) records.push(row);
      row = [];
    } else field += ch;
  }
  row.push(field);
  if (row.some((c) => c !== "")) records.push(row);
  const [header = [], ...data] = records;
  return data.map((r) => Object.fromEntries(header.map((h, i) => [h.trim(), r[i] ?? ""])));
}

export const RISK_STYLES: Record<RiskLevel, string> = {
  High: "bg-danger/15 text-danger",
  Medium: "bg-warning/15 text-warning",
  Low: "bg-success/15 text-success",
};
