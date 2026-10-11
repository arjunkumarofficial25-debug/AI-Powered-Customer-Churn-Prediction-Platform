import type { Metadata } from "next";
import analytics from "@/lib/generated/analytics.json";
import { ChurnBarChart } from "@/components/churn-bar-chart";
import { Card, PageHeader, Stat } from "@/components/ui";
import { pct } from "@/lib/utils";

export const metadata: Metadata = { title: "Analytics | Churn Prediction Platform" };

const SECTIONS = [
  { title: "By contract type", data: analytics.by_contract },
  { title: "By tenure (months)", data: analytics.by_tenure },
  { title: "By payment method", data: analytics.by_payment },
  { title: "By internet service", data: analytics.by_internet },
];

export default function AnalyticsPage() {
  const { kpis } = analytics;
  return (
    <>
      <PageHeader title="Analytics dashboard" description="Churn rates across the full customer base. Segments above the overall churn rate are highlighted in red." />
      <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-4">
        <Stat label="Customers" value={kpis.customers.toLocaleString()} />
        <Stat label="Churn rate" value={pct(kpis.churn_rate)} />
        <Stat label="Avg monthly bill" value={`$${kpis.avg_monthly.toFixed(2)}`} />
        <Stat label="Monthly revenue lost" value={`$${Math.round(kpis.revenue_at_risk).toLocaleString()}`} hint="From churned customers" />
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        {SECTIONS.map((s) => (
          <Card key={s.title}>
            <h2 className="mb-3 font-medium">{s.title}</h2>
            <ChurnBarChart data={s.data} overall={kpis.churn_rate} />
          </Card>
        ))}
      </div>
    </>
  );
}
