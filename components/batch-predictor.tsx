"use client";

import { useState } from "react";
import { Download, Upload } from "lucide-react";
import { Card, Stat } from "@/components/ui";
import { cleanRow, parseCsv, predict, riskLevel, RISK_STYLES, validateRows, type RiskLevel } from "@/lib/model";
import { cn, pct } from "@/lib/utils";

type Scored = Record<string, string> & { churn_probability: string; prediction: string; risk_level: RiskLevel };

const MAX_BYTES = 10 * 1024 * 1024;

function toCsv(rows: Record<string, string>[]) {
  const header = Object.keys(rows[0]);
  const esc = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  return [header.join(","), ...rows.map((r) => header.map((h) => esc(String(r[h] ?? ""))).join(","))].join("\n");
}

export function BatchPredictor() {
  const [results, setResults] = useState<Scored[] | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [fileName, setFileName] = useState("");

  async function handleFile(file: File | undefined) {
    setResults(null);
    setErrors([]);
    if (!file) return;
    setFileName(file.name);
    if (file.size > MAX_BYTES) return setErrors(["File is larger than 10 MB."]);
    const rows = parseCsv(await file.text());
    const problems = validateRows(rows);
    if (problems.length) return setErrors(problems);
    const scored = rows
      .map((r) => {
        const p = predict(cleanRow(r));
        const risk = riskLevel(p);
        return { ...r, churn_probability: p.toFixed(4), prediction: risk === "High" ? "Churn" : "Stay", risk_level: risk };
      })
      .sort((a, b) => Number(b.churn_probability) - Number(a.churn_probability));
    setResults(scored);
  }

  function download() {
    if (!results) return;
    const url = URL.createObjectURL(new Blob([toCsv(results)], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "churn_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  const high = results?.filter((r) => r.risk_level === "High").length ?? 0;

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <label className="flex cursor-pointer flex-col items-center gap-2 rounded-md border border-dashed border-border px-4 py-10 text-center transition-colors hover:bg-muted/50 focus-within:ring-2 focus-within:ring-primary">
          <Upload className="size-6 text-primary" aria-hidden="true" />
          <span className="font-medium">{fileName || "Choose a CSV file"}</span>
          <span className="text-xs text-muted-foreground">Same columns as the Telco dataset. customerID and Churn are optional.</span>
          <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => handleFile(e.target.files?.[0])} />
        </label>
        <p className="mt-3 text-xs text-muted-foreground">
          {"Need a sample? "}
          <a className="text-primary underline-offset-4 hover:underline" href="/sample_customers.csv" download>
            Download 50 example customers
          </a>
          . Files are processed locally and never uploaded.
        </p>
      </Card>

      {errors.length ? (
        <div role="alert" className="rounded-lg border border-danger/40 bg-danger/10 p-4 text-sm text-danger">
          {errors.map((e) => <p key={e}>{e}</p>)}
        </div>
      ) : null}

      {results ? (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Stat label="Customers scored" value={results.length.toLocaleString()} />
            <Stat label="High risk" value={high.toLocaleString()} hint={pct(high / results.length)} />
            <div className="col-span-2 flex items-center md:justify-end">
              <button
                type="button"
                onClick={download}
                className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90"
              >
                <Download className="size-4" aria-hidden="true" />
                Download results
              </button>
            </div>
          </div>
          <Card className="overflow-x-auto p-0">
            <table className="w-full text-sm">
              <thead className="border-b border-border text-left text-xs text-muted-foreground">
                <tr>
                  {["Customer", "Contract", "Tenure", "Monthly", "Probability", "Risk"].map((h) => (
                    <th key={h} scope="col" className="px-4 py-3 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.slice(0, 200).map((r, i) => (
                  <tr key={r.customerID ?? i} className="border-b border-border/60 last:border-0">
                    <td className="px-4 py-2.5 font-mono text-xs">{r.customerID ?? `#${i + 1}`}</td>
                    <td className="px-4 py-2.5">{r.Contract}</td>
                    <td className="px-4 py-2.5 font-mono">{r.tenure}</td>
                    <td className="px-4 py-2.5 font-mono">${r.MonthlyCharges}</td>
                    <td className="px-4 py-2.5 font-mono">{pct(Number(r.churn_probability))}</td>
                    <td className="px-4 py-2.5">
                      <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", RISK_STYLES[r.risk_level])}>{r.risk_level}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {results.length > 200 ? (
              <p className="px-4 py-3 text-xs text-muted-foreground">Showing the 200 highest-risk customers. Download for the full list.</p>
            ) : null}
          </Card>
        </>
      ) : null}
    </div>
  );
}
