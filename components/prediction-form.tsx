"use client";

import { useMemo, useState } from "react";
import { Card } from "@/components/ui";
import { CATEGORICAL, explain, model, predict, recommendations, riskLevel, RISK_STYLES, type Customer } from "@/lib/model";
import { cn, pct } from "@/lib/utils";

const LABELS: Record<string, string> = {
  SeniorCitizen: "Senior citizen",
  tenure: "Tenure (months)",
  MonthlyCharges: "Monthly charges ($)",
  TotalCharges: "Total charges ($)",
  gender: "Gender",
  Partner: "Partner",
  Dependents: "Dependents",
  PhoneService: "Phone service",
  MultipleLines: "Multiple lines",
  InternetService: "Internet service",
  OnlineSecurity: "Online security",
  OnlineBackup: "Online backup",
  DeviceProtection: "Device protection",
  TechSupport: "Tech support",
  StreamingTV: "Streaming TV",
  StreamingMovies: "Streaming movies",
  Contract: "Contract",
  PaperlessBilling: "Paperless billing",
  PaymentMethod: "Payment method",
};

const DEFAULT: Customer = {
  SeniorCitizen: 0, tenure: 5, MonthlyCharges: 85, TotalCharges: 425,
  gender: "Female", Partner: "No", Dependents: "No", PhoneService: "Yes", MultipleLines: "No",
  InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No",
  TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "No", Contract: "Month-to-month",
  PaperlessBilling: "Yes", PaymentMethod: "Electronic check",
};

const fieldClass = "w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary";

export function PredictionForm() {
  const [customer, setCustomer] = useState<Customer>(DEFAULT);
  const set = (key: string, value: string | number) => setCustomer((c) => ({ ...c, [key]: value }));

  const probability = predict(customer);
  const risk = riskLevel(probability);
  const contributions = useMemo(() => explain(customer).slice(0, 8), [customer]);
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.contribution)), 0.01);

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
      <Card>
        <h2 className="mb-4 font-medium">Customer profile</h2>
        <form className="grid gap-4 sm:grid-cols-2" onSubmit={(e) => e.preventDefault()}>
          <label className="flex flex-col gap-1.5 text-sm">
            <span className="text-muted-foreground">{LABELS.SeniorCitizen}</span>
            <select className={fieldClass} value={customer.SeniorCitizen} onChange={(e) => set("SeniorCitizen", Number(e.target.value))}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </label>
          {(["tenure", "MonthlyCharges", "TotalCharges"] as const).map((key) => (
            <label key={key} className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted-foreground">{LABELS[key]}</span>
              <input
                type="number"
                min={0}
                step={key === "tenure" ? 1 : 0.01}
                className={cn(fieldClass, "font-mono")}
                value={customer[key]}
                onChange={(e) => set(key, Number(e.target.value))}
              />
            </label>
          ))}
          {CATEGORICAL.map((key) => (
            <label key={key} className="flex flex-col gap-1.5 text-sm">
              <span className="text-muted-foreground">{LABELS[key]}</span>
              <select className={fieldClass} value={String(customer[key])} onChange={(e) => set(key, e.target.value)}>
                {model.columnOptions[key].map((opt) => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </label>
          ))}
        </form>
      </Card>

      <div className="flex flex-col gap-6">
        <Card aria-live="polite">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm text-muted-foreground">Churn probability</p>
              <p className="font-mono text-5xl font-semibold">{pct(probability)}</p>
            </div>
            <span className={cn("rounded-full px-3 py-1 text-sm font-medium", RISK_STYLES[risk])}>{risk} risk</span>
          </div>
          <div className="mt-5 h-2 w-full overflow-hidden rounded-full bg-muted" role="presentation">
            <div
              className={cn("h-full rounded-full", risk === "High" ? "bg-danger" : risk === "Medium" ? "bg-warning" : "bg-success")}
              style={{ width: `${probability * 100}%` }}
            />
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Decision threshold {pct(model.threshold, 0)} (chosen to maximise F1 on out-of-fold training predictions).
          </p>
        </Card>

        <Card>
          <h2 className="mb-1 font-medium">Why this prediction?</h2>
          <p className="mb-4 text-xs text-muted-foreground">
            Feature contributions in log-odds versus the average customer. Red pushes toward churn, green away.
          </p>
          <ul className="flex flex-col gap-2.5">
            {contributions.map((c) => (
              <li key={c.feature} className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-3 text-sm">
                <span className="truncate" title={`${LABELS[c.feature]}: ${c.value}`}>
                  {LABELS[c.feature]}
                  <span className="block truncate text-xs text-muted-foreground">{String(c.value)}</span>
                </span>
                <div className="relative h-2 rounded-full bg-muted">
                  <div
                    className={cn("absolute top-0 h-full rounded-full", c.contribution > 0 ? "left-1/2 bg-danger" : "right-1/2 bg-success")}
                    style={{ width: `${(Math.abs(c.contribution) / maxAbs) * 50}%` }}
                  />
                </div>
                <span className="text-right font-mono text-xs text-muted-foreground">
                  {c.contribution > 0 ? "+" : ""}{c.contribution.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </Card>

        <Card>
          <h2 className="mb-3 font-medium">Suggested retention actions</h2>
          <ul className="flex list-disc flex-col gap-2 pl-5 text-sm leading-relaxed">
            {recommendations(customer).map((tip) => <li key={tip}>{tip}</li>)}
          </ul>
        </Card>
      </div>
    </div>
  );
}
