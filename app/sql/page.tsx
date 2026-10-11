import type { Metadata } from "next";
import { SqlExplorer } from "@/components/sql-explorer";
import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "SQL explorer | Churn Prediction Platform" };

export default function SqlPage() {
  return (
    <>
      <PageHeader title="SQL explorer" description="Business queries from sql/queries.sql, executed against the customers table in SQLite when the model is exported." />
      <SqlExplorer />
    </>
  );
}
